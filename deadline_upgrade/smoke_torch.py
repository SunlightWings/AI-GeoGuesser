"""CPU-only model/KD checkpoint smoke. Random tensors are tests, never training data."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from geoguesser.model import build_model,model_config_from_runtime
from geoguesser.cells import CountryKMeansPartition
from geoguesser.checkpoint import make_checkpoint_bundle,load_checkpoint
from geoguesser.losses import GeoCriterion
from student import count_check

torch.set_num_threads(2)
cfg=json.loads(Path(__file__).with_name('config.json').read_text())
frame=pd.DataFrame(dict(country_id=[0,0,1,1],lat=[50,51,60,61],lng=[10,11,20,21]))
fine=CountryKMeansPartition(2); coarse=CountryKMeansPartition(4)
fine_ids=fine.fit_transform(frame); coarse_ids=coarse.fit_transform(frame)
mc=model_config_from_runtime(cfg,2,fine.n_cells_,coarse.n_cells_)
model=build_model(mc,load_external_weights=False)
out=model(torch.randn(4,3,64,64))
gps=torch.tensor(frame[['lat','lng']].to_numpy(),dtype=torch.float32)
from geoguesser.geo import latlng_to_unit_torch
batch=dict(latlng=gps,xyz=latlng_to_unit_torch(gps),country_id=torch.tensor(frame.country_id.to_numpy()),fine_cell=torch.tensor(fine_ids),coarse_cell=torch.tensor(coarse_ids))
loss,_=GeoCriterion(fine,coarse,cfg)(out,batch)
temp=2.; target=torch.randn_like(out['fine_logits'])
loss=loss+.3*F.kl_div(F.log_softmax(out['fine_logits']/temp,1),F.softmax(target/temp,1),reduction='batchmean')*temp**2
loss=loss+.3*(1-F.cosine_similarity(out['embedding'],torch.randn_like(out['embedding']),dim=1)).mean()
assert torch.isfinite(loss); loss.backward(); count_check(model)
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)/'model.pt'
    torch.save(make_checkpoint_bundle(model,mc,fine,coarse,{'X':0,'Y':1},cfg,{'image_size':64}),p)
    restored,_,_,_=load_checkpoint(p,torch.device('cpu'))
    assert count_check(restored)==count_check(model)
print('PASS: same-head supervised + probability + feature losses; backward; <=5M; checkpoint reload')
