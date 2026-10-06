"""Time fill using immutable observations; no cross-pipe global fallback."""
import numpy as np
import pandas as pd

def fill_daily(series):
    s=series.sort_index().astype(float)
    if not s.index.is_unique:raise ValueError('Duplicate dates')
    original=s.copy();values=s.to_numpy(copy=True);labels=np.full(len(s),'observed',dtype=object)
    labels[s.isna().to_numpy()]='missing'
    if s.notna().sum()==0:return s,pd.Series(labels,index=s.index)
    dates=s.index;missing=original.isna().to_numpy();j=0
    observed=original.dropna();monthly=observed.groupby([observed.index.year,observed.index.month]).mean()
    mean=float(observed.mean());known=observed.to_dict();choices={}
    while j<len(s):
        if not missing[j]:j+=1;continue
        a=j
        while j<len(s) and missing[j]:j+=1
        b=j
        long_gap=dates[b-1]+pd.Timedelta(days=1)>=dates[a]+pd.DateOffset(months=6)
        if not long_gap and a>0 and b<len(s):
            span=(dates[b]-dates[a-1]).days
            f=(dates[a:b]-dates[a-1]).days.to_numpy()/span
            values[a:b]=original.iloc[a-1]+f*(original.iloc[b]-original.iloc[a-1]);labels[a:b]='interpolation_less_6_months'
        else:
            for k in range(a,b):
                day=dates[k];key=(day.year,day.month)
                if key not in choices:
                    candidates=[(int(y),float(v)) for (y,m),v in monthly.items() if m==day.month and y!=day.year]
                    choices[key]=min(candidates,key=lambda z:(abs(z[0]-day.year),z[0])) if candidates else None
                if choices[key] is not None:
                    y,v=choices[key]
                    try:analog=day.replace(year=y)
                    except ValueError:analog=day.replace(year=y,day=28)
                    values[k]=known.get(analog,v);labels[k]='seasonal_'+str(y)
                else:values[k]=mean;labels[k]='same_pipe_mean_no_analog'
    s=pd.Series(values,index=dates);labels=pd.Series(labels,index=dates)
    assert np.allclose(s[original.notna()],original.dropna(),rtol=0,atol=0)
    return s,labels
