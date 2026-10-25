"""Portable complete evidence in bounded, checksummed archive segments."""
import gzip,io,json,shutil,tarfile
from pathlib import Path
from . import schema
from .manifest import file_hash,write_new
from .pdb import load

class Segments(io.RawIOBase):
 def __init__(self,directory,limit):self.directory=directory;self.limit=limit;self.stream=None;self.size=0;self.paths=[]
 def writable(self):return True
 def write(self,data):
  total=len(data);view=memoryview(data)
  while view:
   if self.stream is None or self.size==self.limit:
    if self.stream:self.stream.close()
    path=self.directory/f'evidence.tar.gz.part{len(self.paths)+1:03d}';self.paths.append(path);self.stream=path.open('xb');self.size=0
   n=min(len(view),self.limit-self.size);self.stream.write(view[:n]);self.size+=n;view=view[n:]
  return total
 def finish(self):
  if self.stream:self.stream.close()

def pack(directory,output,limit=32*1024*1024):
 directory=Path(directory);output=Path(output);manifest,rows=load(directory);output.mkdir(parents=True,exist_ok=False);segments=Segments(output,limit);root=f'posteriordb-v{manifest["schema_version"]}'
 try:
  with gzip.GzipFile(fileobj=segments,mode='wb',filename='',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w|') as archive:
   for path in sorted(directory.rglob('*')):
    if not path.is_file() or path.is_symlink():continue
    item=tarfile.TarInfo(root+'/'+path.relative_to(directory).as_posix());item.size=path.stat().st_size;item.mode=0o644;item.mtime=0
    with path.open('rb') as stream:archive.addfile(item,stream)
 finally:segments.finish()
 for name in ('manifest.json','report.json','report.md','report.html','summary.csv','reference-comparisons.csv'):
  if (directory/name).is_file():shutil.copyfile(directory/name,output/name)
 write_new(output/'bundle.json',{'schema_version':manifest['schema_version'],'manifest_sha256':manifest['manifest_sha256'],'planned_jobs':len(rows),'root':root,'parts':[{'name':p.name,'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in segments.paths],'review_files':{p.name:file_hash(p) for p in output.iterdir() if p.is_file() and p not in segments.paths}})
 return output/'bundle.json'

def unpack(bundle,output):
 bundle=Path(bundle);output=Path(output);index=schema.load(bundle/'bundle.json');schema.keys(index,{'schema_version','manifest_sha256','planned_jobs','root','parts','review_files'},'evidence bundle')
 if type(index['schema_version']) is not int or index['schema_version'] not in (2,3) or index['root']!=f'posteriordb-v{index["schema_version"]}' or not 1<=len(index['parts'])<=256:raise schema.Invalid('invalid evidence bundle')
 for i,part in enumerate(index['parts']):
  schema.keys(part,{'name','bytes','sha256'},'archive part')
  if part['name']!=f'evidence.tar.gz.part{i+1:03d}' or not 0<part['bytes']<=64*1024*1024:raise schema.Invalid('invalid archive segment')
  path=bundle/part['name']
  if path.stat().st_size!=part['bytes'] or file_hash(path)!=part['sha256']:raise schema.Invalid('archive segment drift')
 for name,sha in index['review_files'].items():
  schema.relative(name)
  if file_hash(bundle/name)!=sha:raise schema.Invalid('review file drift')
 output.mkdir(parents=True,exist_ok=False);combined=output/'archive.tar.gz'
 with combined.open('xb') as stream:
  for part in index['parts']:
   with (bundle/part['name']).open('rb') as source:shutil.copyfileobj(source,stream)
 with tarfile.open(combined,'r:gz') as archive:
  for item in archive.getmembers():
   if not item.isfile() or not item.name.startswith(index['root']+'/') or item.size>1024*1024*1024:raise schema.Invalid('invalid evidence archive member')
   schema.relative(item.name)
  archive.extractall(output,filter='data')
 combined.unlink();root=output/index['root'];manifest,rows=load(root)
 if manifest['manifest_sha256']!=index['manifest_sha256'] or len(rows)!=index['planned_jobs']:raise schema.Invalid('bundle identity drift')
 return root
