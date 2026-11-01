"""Парсинг ежедневных официальных курсов USD Нацбанка РБ"""
import glob, re, sys
import pandas as pd, xlrd

def parse_file(path):
    wb = xlrd.open_workbook(path, logfile=open("/dev/null", "w"))
    s = wb.sheet_by_index(0)
    hdr = None
    for r in range(0, 8):
        row = [str(c) for c in s.row_values(r)]
        if row and row[0].strip() == "Дата":
            hdr = r; break
    row = [str(c) for c in s.row_values(hdr)]
    col = next(i for i, c in enumerate(row) if "(USD)" in c)
    two_cols = (hdr + 1 < s.nrows and "Кол-во" in " ".join(str(c) for c in s.row_values(hdr + 1)))
    start = hdr + 2 if two_cols else hdr + 1
    out = []
    for r in range(start, s.nrows):
        d = s.cell_value(r, 0)
        if not isinstance(d, float): continue
        date = xlrd.xldate_as_datetime(d, wb.datemode)
        if two_cols:
            q, v = s.cell_value(r, col), s.cell_value(r, col + 1)
            if q == "" or v == "": continue
            rate = float(v) / float(q)
        else:
            v = s.cell_value(r, col)
            if v == "": continue
            rate = float(v)
        out.append((date, rate))
    return pd.DataFrame(out, columns=["date", "usd_raw"])

def load_all(pattern):
    df = pd.concat([parse_file(f) for f in sorted(glob.glob(pattern))]).drop_duplicates("date").sort_values("date")
    
    df["byn_per_usd"] = df["usd_raw"].where(df["usd_raw"] < 100, df["usd_raw"] / 10000)
    df["year"] = df["date"].dt.year
    return df

if __name__ == "__main__":
    df = load_all(sys.argv[1])
    df.to_csv(sys.argv[2], index=False)
    ann = df.groupby("year")["byn_per_usd"].agg(["mean", "min", "max", "count"]).round(4)
    print(ann.to_string())
    print("\n2016 mixed raw max/min:", df[df.year==2016].usd_raw.agg(['min','max']).tolist())
