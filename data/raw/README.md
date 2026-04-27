# Dataset Files

Place your downloaded dataset files here.

## Required Files

| Filename | Download From | Notes |
|----------|--------------|-------|
| `symptom2disease.csv` | https://www.kaggle.com/datasets/niyarrbarman/symptom2disease | Direct download |
| `disease_description.csv` | https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset | From zip |
| `disease_precaution.csv` | Same as above | From zip |
| `medquad.json` | https://github.com/abachaa/MedQuAD | See conversion note below |
| `medmcqa_train.jsonl` | https://huggingface.co/datasets/medmcqa | Download train split |

## MedQuAD Conversion

MedQuAD comes as XML files. Convert to JSON:
```bash
# After cloning the repo, run this Python script:
python3 -c "
import os, json, xml.etree.ElementTree as ET
results = []
for root_dir, dirs, files in os.walk('MedQuAD'):
    for f in files:
        if f.endswith('.xml'):
            try:
                tree = ET.parse(os.path.join(root_dir, f))
                root = tree.getroot()
                focus = root.find('.//Focus')
                for qa in root.findall('.//QAPair'):
                    q = qa.find('Question')
                    a = qa.find('Answer')
                    if q is not None and a is not None:
                        results.append({'focus': focus.text if focus is not None else '', 'question': q.text or '', 'answer': a.text or ''})
            except: pass
with open('medquad.json', 'w') as out:
    json.dump(results, out)
print(f'Converted {len(results)} QA pairs')
"
```

## Notes
- The built-in knowledge base works without any of these files
- Start with `symptom2disease.csv` and `disease_description.csv` — easiest to get
- MedMCQA is large (190k rows) — the ingest script caps at 20k rows
