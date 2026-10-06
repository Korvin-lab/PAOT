"""Launch the complete strict PE2 chain into a new timestamped run directory."""
from pathlib import Path
from datetime import datetime
import subprocess,sys,json
import shutil
ROOT=Path(__file__).resolve().parent
LOG_DIR=None

def run(script,*args):
 command=[sys.executable,str(ROOT/script),*map(str,args)]
 print('RUN:',subprocess.list2cmdline(command),flush=True)
 if LOG_DIR is None:
  subprocess.run(command,cwd=ROOT,check=True)
 else:
  with (LOG_DIR/(Path(script).stem+'.log')).open('a',encoding='utf-8') as log:
   process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                            text=True,encoding='utf-8',errors='replace')
   for line in process.stdout:
    print(line,end='',flush=True);log.write(line);log.flush()
   code=process.wait()
   if code:raise subprocess.CalledProcessError(code,command)

def main():
 global LOG_DIR
 shutil.copy2(ROOT/'input/graph_orenburg_requested_daily_params__SOURCE_84.json',ROOT/'input/graph_orenburg_requested_daily_params.json')
 shutil.copy2(ROOT/'input/graph_orenburg_master__SOURCE_84.json',ROOT/'input/graph_orenburg_master.json')
 run('rebuild_chemistry_inputs.py')
 run('clean_two_phase_chemistry_base.py')
 run('prepare_auxiliary_series.py')
 run('preflight_84.py')
 run('recover_full84_inputs.py','--requested-json',ROOT/'input/graph_orenburg_requested_daily_params.json','--exclusions-csv',ROOT/'input/excluded_active_dates_forecast.csv','--report-json',ROOT/'FULL84_RECOVERY_REPORT.json','--changes-csv',ROOT/'FULL84_RECOVERY_CHANGES.csv')
 run('preflight_84.py','--require-ready')
 out=ROOT/('RUN_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
 out.mkdir(exist_ok=False);strict=out/'strict';stage=out/'stage1';final=out/'final';strict.mkdir();stage.mkdir();final.mkdir()
 LOG_DIR=out/'logs';LOG_DIR.mkdir()
 run('main_pipeline_final_csv.py','--master-json',ROOT/'input/graph_orenburg_master.json','--requested-json',ROOT/'input/graph_orenburg_requested_daily_params.json','--pe2-dll',ROOT/'deps/pe_2_main.dll','--out-dir',strict,'--pe2-trace-csv','NUL','--calc-mode','strict')
 run('create_ascii_aliases.py','--var-out',strict,'--stage1-out',stage,'--chem-out',out/'unused_chem_alias')
 seg=strict/'segments_all_pipes.csv'
 run('validate_temperature_segments.py','--segments-csv',seg,'--master-json',ROOT/'input/graph_orenburg_master.json','--step-m','10','--temperature-min','5','--temperature-max','90','--floor-epsilon','0.05','--max-floor-share-per-profile','0.05','--max-floor-share-total','0.005','--report-json',strict/'TEMPERATURE_AND_SEGMENTS_VALIDATION.json','--errors-csv',strict/'TEMPERATURE_AND_SEGMENTS_ERRORS.csv')
 run('build_final_dataset_stage1_co2.py','--segments-csv',seg,'--chem-daily-csv',ROOT/'input/chem_daily_orenburg.csv','--kvch-daily-csv',ROOT/'input/kvch_daily_orenburg.csv','--visc-csv',ROOT/'input/visc_daily_orenburg.csv','--requested-json',ROOT/'input/graph_orenburg_requested_daily_params.json','--dosage-xls',ROOT/'input/dosage_orenburg_normalized.xlsx','--out-dir',stage)
 run('create_ascii_aliases.py','--var-out',strict,'--stage1-out',stage,'--chem-out',out/'unused_chem_alias')
 run('finalize_prepared_dataset.py','--source',stage/'stage1_required_columns.csv','--destination',final/'final_dataset.csv')
 run('add_h2s_gas_phase.py','--source',final/'final_dataset.csv','--destination',final/'final_dataset__WITH_H2S_GAS_PHASE.csv')
 run('audit_final_quality_strict.py','--final-csv',final/'final_dataset__WITH_H2S_GAS_PHASE.csv','--report-json',final/'FINAL_QUALITY_STRICT_AUDIT.json','--temperature-min','5','--floor-epsilon','0.05','--max-floor-share-total','0.005')
 print('DONE:',final/'final_dataset__WITH_H2S_GAS_PHASE.csv',flush=True)
 print('Read FINAL_VALIDATION.json: missing auxiliary values are not fabricated.',flush=True)
if __name__=='__main__':main()
