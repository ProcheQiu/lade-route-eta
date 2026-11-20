"""Record-level cleaning of the LaDe Shanghai files.

Rules R1-R8 drop rows whose label or position cannot be trusted; G1-G2 only blank a bad GPS fix.
Counts per rule are returned so they can be written to results/cleaning_log.json.
Verified on the Shanghai files: keeps 1,474,505 delivery rows (99.37%) and 1,413,208 pickup rows (99.21%).
"""
import numpy as np
import pandas as pd

from . import config as C


def _t(s):
    return pd.to_datetime(f"{C.YEAR}-" + s, format="%Y-%m-%d %H:%M:%S", errors="coerce")


def _inbox(lng, lat):
    b = C.SHANGHAI_BOX
    return lng.between(*b["lng"]) & lat.between(*b["lat"])


def km(lng1, lat1, lng2, lat2):
    """Equirectangular distance in km; accurate to <0.1% at city scale."""
    p = np.radians(lat1)
    return 6371.0 * np.sqrt(np.radians(lat2 - lat1) ** 2 + (np.cos(p) * np.radians(lng2 - lng1)) ** 2)


def clean(raw: pd.DataFrame, name: str):
    """Return (clean_df, log). `name` is 'delivery' or 'pickup'."""
    ev = name
    df = raw.copy()
    log = {"raw_rows": len(df)}

    def drop(mask, rule):
        nonlocal df
        log[rule] = int(mask.sum())
        df = df[~mask].copy()

    tcols = ["accept_time", f"{ev}_time", "accept_gps_time", f"{ev}_gps_time"]
    if name == "pickup":
        tcols += ["time_window_start", "time_window_end"]
    for c in tcols:
        df[c] = _t(df[c])
    df["ds_date"] = pd.to_datetime(f"{C.YEAR}" + df.ds.astype(str).str.zfill(4), format="%Y%m%d")
    df = df.rename(columns={f"{ev}_time": "finish_time", f"{ev}_gps_time": "finish_gps_time",
                            f"{ev}_gps_lng": "finish_gps_lng", f"{ev}_gps_lat": "finish_gps_lat"})

    drop(df.duplicated("order_id"), "R1_duplicate_order_id")
    drop(df.accept_time.isna() | df.finish_time.isna(), "R2_unparseable_time")
    drop(~_inbox(df.lng, df.lat), "R3_stop_outside_shanghai")
    drop(df.finish_time.dt.normalize() != df.ds_date, "R4_finish_not_on_ds")
    drop(df.finish_time < df.accept_time, "R5_finish_before_accept")
    drop(df.finish_time == df.accept_time, "R6_zero_duration")
    if name == "pickup":
        drop(df.time_window_end <= df.time_window_start, "R7_bad_window")
        drop(df.time_window_start.dt.normalize() != df.ds_date, "R8_window_not_on_ds")

    bad_f = df.finish_gps_lng.notna() & (~_inbox(df.finish_gps_lng, df.finish_gps_lat)
                                         | (km(df.lng, df.lat, df.finish_gps_lng, df.finish_gps_lat) > C.GPS_MAX_KM))
    bad_a = df.accept_gps_lng.notna() & ~_inbox(df.accept_gps_lng, df.accept_gps_lat)
    log["G1_finish_gps_nulled"] = int(bad_f.sum())
    log["G2_accept_gps_nulled"] = int(bad_a.sum())
    df.loc[bad_f, ["finish_gps_lng", "finish_gps_lat", "finish_gps_time"]] = np.nan
    df.loc[bad_a, ["accept_gps_lng", "accept_gps_lat", "accept_gps_time"]] = np.nan
    df["finish_gps_missing"] = df.finish_gps_lng.isna()
    df["accept_gps_missing"] = df.accept_gps_lng.isna()

    df["dur_min"] = (df.finish_time - df.accept_time).dt.total_seconds() / 60
    df["finish_hour"] = df.finish_time.dt.hour
    df["month"] = df.ds_date.dt.month
    if name == "pickup":
        df["late"] = df.finish_time > df.time_window_end
        df["early"] = df.finish_time < df.time_window_start
    df = df.sort_values(["courier_id", "ds", "finish_time"]).reset_index(drop=True)
    df["visit_rank"] = df.groupby(["courier_id", "ds"]).cumcount() + 1
    log["clean_rows"] = len(df)
    log["kept_share"] = round(len(df) / log["raw_rows"], 4)
    return df, log


def split_of(ds_date: pd.Series, name: str) -> pd.Series:
    """'train' / 'val' / 'test' by the day-based chronological cut in config."""
    a, b = C.SPLIT_CUTS[name]
    md = ds_date.dt.strftime("%m-%d")
    return pd.Series(np.where(md <= a, "train", np.where(md <= b, "val", "test")), index=ds_date.index)
