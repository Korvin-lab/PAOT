import unittest
import pandas as pd
import numpy as np
from fill_rules import fill_daily
import main_pipeline_final_csv as pipeline

class Rules(unittest.TestCase):
 def test_stage1_iso_dates(self):
  from build_final_dataset_stage1_co2 import normalize_date
  self.assertEqual(normalize_date('2020-03-01'),'2020-03-01')
  self.assertEqual(normalize_date('2020-01-12'),'2020-01-12')
  self.assertEqual(normalize_date('01.03.2020'),'2020-03-01')
  self.assertEqual(normalize_date('2020-03-01T12:30:00'),'2020-03-01')
  self.assertEqual(normalize_date('2020-02-30'),'')
 def test_internal_interpolation_preserves_observed(self):
  s=pd.Series([10,np.nan,np.nan,40],index=pd.date_range('2020-01-01',periods=4))
  out,tags=fill_daily(s);self.assertEqual(out.tolist(),[10,20,30,40]);self.assertEqual(tags.iloc[0],'observed')
 def test_half_year_not_linear_bridge(self):
  dates=pd.date_range('2020-01-01','2021-07-01');s=pd.Series(np.nan,index=dates)
  s.loc['2020-01-01']=12;s.loc['2020-06-15']=18;s.loc['2020-12-31']=1;s.loc['2021-07-01']=80
  out,tags=fill_daily(s)
  self.assertEqual(out.loc['2021-01-01'],12);self.assertTrue(tags.loc['2021-01-01'].startswith('seasonal'))
 def test_all_missing_stays_unknown(self):
  s=pd.Series(np.nan,index=pd.date_range('2020-01-01',periods=10));out,_=fill_daily(s);self.assertTrue(out.isna().all())
 def test_measured_zero_not_replaced(self):
  s=pd.Series([0,np.nan,2],index=pd.date_range('2020-01-01',periods=3));out,_=fill_daily(s);self.assertEqual(out.tolist(),[0,1,2])
 def test_co2_positive_source_rule(self):
  import build_final_dataset_stage1_co2 as stage1
  t=np.array([20.,20.]); co2=np.array([0.,12.]); sal=np.array([150.,150.])
  p,co2_mol=stage1.calculate_pco2_and_mol(t,co2,sal)
  self.assertTrue(np.isnan(p[0]));self.assertTrue(np.isnan(co2_mol[0]))
  self.assertTrue(np.isfinite(p[1]));self.assertTrue(np.isfinite(co2_mol[1]))
 def test_duplicates_rejected(self):
  s=pd.Series([1,2],index=[pd.Timestamp('2020-01-01')]*2)
  with self.assertRaises(ValueError):fill_daily(s)
 def test_boundary_sources(self):
  self.assertEqual(pipeline.classify_pressure_boundary('АГЗУ АСОДУ'),'start')
  self.assertEqual(pipeline.classify_pressure_boundary('трубный техрежим: P факт конец'),'end')
  self.assertEqual(pipeline.classify_temperature_boundary('граф от предшественника: донор 1'),'start')
  self.assertEqual(pipeline.classify_temperature_boundary('трубный техрежим: T конец'),'end')
class FinalChemistry(unittest.TestCase):
 def test_h2s_water_comes_from_two_phase_file_not_conversion(self):
  import tempfile,json
  from pathlib import Path
  import finalize_prepared_dataset as f
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'input').mkdir();old=f.ROOT;f.ROOT=root
   try:
    (root/'run_config.json').write_text(json.dumps({'expected_pipe_ids':['1']}))
    pd.DataFrame({'id простого участка':['1','1'],'Дата контроля':['2020-03-01','2020-03-02'],'CO2':[12.,12.],'Общая минерализация, г/л':[20.,20.],'pH':[7.,7.],'H2S_UKK_source':[3000.,3000.]}).to_csv(root/'input/chem_daily_orenburg.csv',index=False)
    pd.DataFrame({'id':['1','1','1'],'date':['2020-03-01','2020-03-02','2020-03-03'],'H2S in Water Phase':[5.,0.,9.]}).to_csv(root/'input/h2s_two_phase_daily_base.csv',index=False)
    pd.DataFrame({'id':['1'],'date':['2020-03-01'],'segment_id':[0],'CO2 in Water Phase':[12.],'Min':[20.],'pH':[7.],'seg_temperature':[15.],'seg_p_start':[20.],'kvch':[np.nan]}).to_csv(root/'in.csv',index=False)
    f.finalize(root/'in.csv',root/'final/out.csv')
    out=pd.read_csv(root/'final/out.csv');self.assertEqual(out['H2S in Water Phase'].iloc[0],5.);self.assertTrue(pd.isna(out.kvch.iloc[0]));self.assertEqual(out.date.iloc[0],'2020-03-01')
    pd.DataFrame({'id':['1'],'date':['2020-03-02'],'segment_id':[0],'CO2 in Water Phase':[12.],'Min':[20.],'pH':[7.],'seg_temperature':[15.],'seg_p_start':[20.],'kvch':[np.nan]}).to_csv(root/'in_mid.csv',index=False)
    f.finalize(root/'in_mid.csv',root/'final/out_mid.csv')
    out_mid=pd.read_csv(root/'final/out_mid.csv');self.assertEqual(out_mid['H2S in Water Phase'].iloc[0],7.)
   finally:f.ROOT=old

if __name__=='__main__':unittest.main()
