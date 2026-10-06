import hashlib,subprocess,tempfile
from pathlib import Path
here=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as d:
 p=Path(d)
 head=b'\xef\xbb\xbfdate,id,'+b','.join(f'c{i}'.encode() for i in range(2,51))+b'\n'
 rows=[]
 for day,pid in [('2020-01-01','1750004707'),('2020-01-02','1750004707'),('2020-01-02','1750006836')]:
  rows.append((','.join([day,pid]+[str(i) for i in range(2,51)])+'\n').encode())
 source=head+b''.join(rows);(p/'source.csv').write_bytes(source)
 (p/'patch.csv').write_text('id,date,Min\n1750004707,2020-01-01,27.5\n1750006836,2020-01-02,31.25\n')
 sha=hashlib.sha256(source).hexdigest()
 for stem in ['patch_min','verify_min']:
  subprocess.run(['cc','-O3','-std=c11','-DROWS=3','-DKEYS=2','-o',str(p/stem),str(here/(stem+'.c'))],check=True)
 args=[str(p/'patch.csv'),str(p/'source.csv'),str(p/'output.csv'),sha]
 assert 'PASS rows=3' in subprocess.run([str(p/'patch_min'),*args],check=True,capture_output=True,text=True).stdout
 assert 'PASS rows=3' in subprocess.run([str(p/'verify_min'),*args],check=True,capture_output=True,text=True).stdout
 old=[x.decode().strip().split(',') for x in rows]
 new=[x.decode().strip().split(',') for x in (p/'output.csv').read_bytes().splitlines(keepends=True)[1:]]
 assert [r[9] for r in new]==['27.5','9','31.25']
 assert all(a[:9]+a[10:]==b[:9]+b[10:] for a,b in zip(old,new))
 print('PASS small 51-column fixture, only Min changed, SHA and independent verifier')
