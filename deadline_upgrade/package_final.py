"""Package a single full-data student, exact holdout CSV, source and run provenance."""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import torch
from common import digest, validate_submission
from geoguesser.checkpoint import load_checkpoint
from student import count_check

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); p.add_argument('--plan',required=True); p.add_argument('--data',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    run=Path(a.run); root=Path(__file__).resolve().parents[1]
    checkpoint=run/'final_model.pt'; csv=run/'predictions.csv'
    model,_,_,bundle=load_checkpoint(checkpoint,torch.device('cpu'))
    n=count_check(model)
    if bundle['metadata'].get('fold')!='full': raise ValueError('Not a full-data checkpoint')
    if digest(Path(a.plan)/'plan.pt')!=bundle['metadata']['plan_sha256']: raise ValueError('Wrong full-data plan')
    validate_submission(pd.read_csv(csv),Path(a.data)/'holdout_public')
    dest=Path(a.out)
    if dest.exists(): raise ValueError('Submission ZIP already exists; choose a new name')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('final_model.pt','predictions.csv','config.json','run.json','history.csv'):
            z.write(run/name,name)
        for name in ('plan.pt','manifest.json','train_rows.csv','validation_rows.csv'):
            z.write(Path(a.plan)/name,'full_plan/'+name)
        for folder,glob in [('src/geoguesser','*.py'),('deadline_upgrade','*.py'),('deadline_upgrade','*.json'),('deadline_upgrade','*.txt')]:
            for source in sorted((root/folder).glob(glob)): z.write(source,source.relative_to(root))
        z.write(root/'pyproject.toml','pyproject.toml'); z.write(root/'requirements.txt','requirements.txt')
        z.writestr('environment.txt',subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        z.writestr('README_SUBMISSION.md',f'''# Submission\n\nInference neural parameters: {n:,} (all Parameters including frozen).\nTA-approved ImageNet initialization; teachers used only at training time.\nNo teacher or reference bank is required for inference.\n\nInstall matching CUDA PyTorch, then `python -m pip install -e .`.\nPredict: `python deadline_upgrade/student.py predict --checkpoint final_model.pt --data PATH_TO_GEO_DATASET --out reproduced_predictions.csv`\nValidate: `python deadline_upgrade/evaluate.py validate --csv reproduced_predictions.csv --data PATH_TO_GEO_DATASET`\nTraining configuration and exact full-data plan are included. See deadline_upgrade entry points.\nAdd your course-required 1--2 page report before submission.\n''')
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None: raise ValueError('ZIP CRC failed')
    print(json.dumps(dict(zip=str(dest),neural_parameters=n,checkpoint_sha256=digest(checkpoint),predictions_sha256=digest(csv)),indent=2))

if __name__=='__main__': main()
