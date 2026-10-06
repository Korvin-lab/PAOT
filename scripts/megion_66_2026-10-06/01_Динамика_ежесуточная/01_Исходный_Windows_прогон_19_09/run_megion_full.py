"""Run the prepared strict Megion PE2 chain on Windows x64."""
from datetime import datetime
from pathlib import Path
import subprocess, sys

ROOT=Path(__file__).resolve().parent
LOG_DIR=None
def run(script,*args):
 command=[sys.executable,str(ROOT/script),*map(str,args)]
 print('RUN:',subprocess.list2cmdline(command),flush=True)
 if LOG_DIR is None: subprocess.run(command,cwd=ROOT,check=True); return
 with (LOG_DIR/(Path(script).stem+'.log')).open('a',encoding='utf-8') as log:
  p=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
  for line in p.stdout: print(line,end='',flush=True);log.write(line);log.flush()
  if p.wait(): raise subprocess.CalledProcessError(p.returncode,command)
def main():
 global LOG_DIR
 run('preflight_megion.py')
 out=ROOT/('RUN_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')); out.mkdir(exist_ok=False)
 strict=out/'strict'; stage=out/'stage1'; final=out/'final'
 for p in (strict,stage,final):p.mkdir()
 LOG_DIR=out/'logs';LOG_DIR.mkdir()
 run('main_pipeline_final_csv.py','--master-json',ROOT/'input/graph_megion_master.json','--requested-json',ROOT/'input/graph_megion_requested_daily_params.json','--pe2-dll',ROOT/'deps/pe_2_main.dll','--out-dir',strict,'--pe2-trace-csv','NUL','--calc-mode','strict')
 run('validate_temperature_segments.py','--segments-csv',strict/'segments_all_pipes.csv','--master-json',ROOT/'input/graph_megion_master.json','--step-m','10','--temperature-min','5','--temperature-max','90','--floor-epsilon','0.05','--max-floor-share-per-profile','0.05','--max-floor-share-total','0.005','--report-json',strict/'TEMPERATURE_AND_SEGMENTS_VALIDATION.json','--errors-csv',strict/'TEMPERATURE_AND_SEGMENTS_ERRORS.csv')
 run('build_final_dataset_stage1_co2.py','--segments-csv',strict/'segments_all_pipes.csv','--chem-daily-csv',ROOT/'input/chem_daily_megion.csv','--kvch-daily-csv',ROOT/'input/kvch_daily_megion.csv','--visc-csv',ROOT/'input/visc_daily_megion.csv','--requested-json',ROOT/'input/graph_megion_requested_daily_params.json','--out-dir',stage)
 run('create_ascii_aliases.py','--var-out',strict,'--stage1-out',stage,'--chem-out',out/'unused_chem_alias')
 run('finalize_prepared_dataset.py','--source',stage/'stage1_required_columns.csv','--destination',final/'final_dataset.csv','--chem-daily-csv',ROOT/'input/chem_daily_megion.csv')
 run('add_h2s_gas_phase.py','--source',final/'final_dataset.csv','--destination',final/'final_dataset__WITH_H2S_GAS_PHASE.csv')
 run('audit_final_quality_strict.py','--final-csv',final/'final_dataset__WITH_H2S_GAS_PHASE.csv','--report-json',final/'FINAL_QUALITY_STRICT_AUDIT.json','--temperature-min','5','--floor-epsilon','0.05','--max-floor-share-total','0.005')
 print('DONE:',final/'final_dataset__WITH_H2S_GAS_PHASE.csv',flush=True)
if __name__=='__main__':main()
