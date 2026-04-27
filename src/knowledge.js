/**
 * knowledge.js
 * Loads all datasets into memory at startup.
 * No database, no API keys needed.
 *
 * Datasets loaded:
 *   - project/Symptom2Disease.csv
 *   - project/DSP/dataset.csv
 *   - project/DSP/symptom_Description.csv
 *   - project/DSP/symptom_precaution.csv
 *   - project/DSP/Symptom-severity.csv
 *   - project/MedQuAD-master/ (all XML files)
 */

import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import { parseStringPromise } from 'xml2js';

const PROJECT_DIR = path.resolve(process.cwd(), '..', 'project');
const MEDQUAD_DIR = path.join(PROJECT_DIR, 'MedQuAD-master');
const DSP_DIR     = path.join(PROJECT_DIR, 'DSP');
const S2D_FILE    = path.join(PROJECT_DIR, 'Symptom2Disease.csv');

// ─── KNOWLEDGE STORE ─────────────────────────────────────────────────────────
// Each entry: { title, symptoms[], description, precautions[], source, category }

export const knowledgeBase = {
  diseases: [],   // { name, symptoms[], description, precautions[], severity }
  qa: [],         // { focus, question, answer }
  symptomSeverity: {}, // symptom → weight
};

function clean(t) {
  return (t || '').replace(/\s+/g, ' ').replace(/_/g, ' ').trim().toLowerCase();
}

// ─── LOADERS ─────────────────────────────────────────────────────────────────

function loadDSP() {
  const diseaseMap = {}; // name → { symptoms, description, precautions }

  // dataset.csv — symptoms per disease
  const datasetFile = path.join(DSP_DIR, 'dataset.csv');
  if (fs.existsSync(datasetFile)) {
    const rows = parse(fs.readFileSync(datasetFile, 'utf8'), {
      columns: true, skip_empty_lines: true, trim: true,
    });
    rows.forEach(row => {
      const name = (row.Disease || '').trim();
      if (!name) return;
      if (!diseaseMap[name]) diseaseMap[name] = { symptoms: new Set(), description: '', precautions: [] };
      Object.values(row).forEach(v => {
        const s = clean(v);
        if (s && s !== clean(name)) diseaseMap[name].symptoms.add(s);
      });
    });
    console.log(`  DSP dataset.csv: ${Object.keys(diseaseMap).length} diseases`);
  }

  // symptom_Description.csv
  const descFile = path.join(DSP_DIR, 'symptom_Description.csv');
  if (fs.existsSync(descFile)) {
    const rows = parse(fs.readFileSync(descFile, 'utf8'), {
      columns: true, skip_empty_lines: true, trim: true,
    });
    rows.forEach(row => {
      const name = (row.Disease || '').trim();
      const desc = (row.Description || '').trim();
      if (!name || !desc) return;
      if (!diseaseMap[name]) diseaseMap[name] = { symptoms: new Set(), description: '', precautions: [] };
      diseaseMap[name].description = desc;
    });
    console.log(`  DSP symptom_Description.csv: ${rows.length} descriptions`);
  }

  // symptom_precaution.csv
  const precFile = path.join(DSP_DIR, 'symptom_precaution.csv');
  if (fs.existsSync(precFile)) {
    const rows = parse(fs.readFileSync(precFile, 'utf8'), {
      columns: true, skip_empty_lines: true, trim: true,
    });
    rows.forEach(row => {
      const name = (row.Disease || '').trim();
      const precs = [row.Precaution_1, row.Precaution_2, row.Precaution_3, row.Precaution_4]
        .map(p => (p || '').trim()).filter(Boolean);
      if (!name) return;
      if (!diseaseMap[name]) diseaseMap[name] = { symptoms: new Set(), description: '', precautions: [] };
      diseaseMap[name].precautions = precs;
    });
    console.log(`  DSP symptom_precaution.csv: ${rows.length} precautions`);
  }

  // Symptom-severity.csv
  const sevFile = path.join(DSP_DIR, 'Symptom-severity.csv');
  if (fs.existsSync(sevFile)) {
    const rows = parse(fs.readFileSync(sevFile, 'utf8'), {
      columns: true, skip_empty_lines: true, trim: true,
    });
    rows.forEach(row => {
      const sym = clean(row.Symptom || row.symptom || '');
      const w   = parseInt(row.weight || row.Weight || '0', 10);
      if (sym) knowledgeBase.symptomSeverity[sym] = w;
    });
    console.log(`  DSP Symptom-severity.csv: ${rows.length} entries`);
  }

  // Convert to array
  Object.entries(diseaseMap).forEach(([name, data]) => {
    knowledgeBase.diseases.push({
      name,
      nameLower: name.toLowerCase(),
      symptoms: [...data.symptoms],
      description: data.description,
      precautions: data.precautions,
      source: 'DSP',
    });
  });
}

function loadSymptom2Disease() {
  if (!fs.existsSync(S2D_FILE)) return;

  const rows = parse(fs.readFileSync(S2D_FILE, 'utf8'), {
    columns: true, skip_empty_lines: true, trim: true,
  });

  // Group by disease, collect symptom text
  const diseaseMap = {};
  rows.forEach(row => {
    const label = (row.label || '').trim();
    const text  = (row.text || '').trim().toLowerCase();
    if (!label || !text) return;
    if (!diseaseMap[label]) diseaseMap[label] = { texts: [], symptoms: new Set() };
    diseaseMap[label].texts.push(text);
    // Extract keywords from text as symptoms
    text.split(/[\s,\.]+/).forEach(w => {
      if (w.length > 4) diseaseMap[label].symptoms.add(w);
    });
  });

  let added = 0;
  Object.entries(diseaseMap).forEach(([name, data]) => {
    // Only add if not already in KB from DSP
    const exists = knowledgeBase.diseases.find(d => d.nameLower === name.toLowerCase());
    if (!exists) {
      knowledgeBase.diseases.push({
        name,
        nameLower: name.toLowerCase(),
        symptoms: [...data.symptoms].slice(0, 30),
        description: data.texts[0] || '',
        precautions: [],
        source: 'Symptom2Disease',
      });
      added++;
    } else {
      // Merge symptoms
      data.symptoms.forEach(s => {
        if (!exists.symptoms.includes(s)) exists.symptoms.push(s);
      });
    }
  });
  console.log(`  Symptom2Disease.csv: ${rows.length} rows, ${added} new diseases added`);
}

async function loadMedQuAD() {
  if (!fs.existsSync(MEDQUAD_DIR)) return;

  const subfolders = fs.readdirSync(MEDQUAD_DIR)
    .filter(d => fs.statSync(path.join(MEDQUAD_DIR, d)).isDirectory());

  let fileCount = 0;
  let qaCount = 0;

  for (const folder of subfolders) {
    const folderPath = path.join(MEDQUAD_DIR, folder);
    const xmlFiles = fs.readdirSync(folderPath).filter(f => f.endsWith('.xml'));

    for (const xmlFile of xmlFiles) {
      try {
        const xml = fs.readFileSync(path.join(folderPath, xmlFile), 'utf8');
        const doc = await parseStringPromise(xml, { explicitArray: false });
        const focus = (doc?.Document?.Focus || '').trim();
        const qaPairs = doc?.Document?.QAPairs?.QAPair;
        if (!qaPairs) continue;

        const pairs = Array.isArray(qaPairs) ? qaPairs : [qaPairs];
        pairs.forEach(pair => {
          const q = (pair?.Question?._ || pair?.Question || '').replace(/\s+/g, ' ').trim();
          const a = (pair?.Answer || '').replace(/\s+/g, ' ').trim();
          if (q && a && a.length > 40) {
            knowledgeBase.qa.push({
              focus,
              focusLower: focus.toLowerCase(),
              question: q,
              questionLower: q.toLowerCase(),
              answer: a.slice(0, 1200), // cap length
              qtype: pair?.Question?.$?.qtype || '',
            });
            qaCount++;
          }
        });
        fileCount++;
      } catch (_) {}
    }
  }
  console.log(`  MedQuAD: ${fileCount} files → ${qaCount} Q&A pairs`);
}

// ─── BUILT-IN FALLBACK ────────────────────────────────────────────────────────

function loadBuiltin() {
  const builtin = [
    { name: 'Hypertension', symptoms: ['headache', 'dizziness', 'blurred vision', 'chest pain', 'shortness of breath', 'nosebleed'], description: 'High blood pressure (above 130/80 mmHg). Often called the silent killer as it may have no symptoms. Risk factors: obesity, smoking, high salt diet, stress, family history.', precautions: ['Reduce salt intake', 'Exercise regularly', 'Avoid smoking and alcohol', 'Monitor blood pressure daily', 'Take prescribed medications'] },
    { name: 'Type 2 Diabetes', symptoms: ['increased thirst', 'frequent urination', 'fatigue', 'blurred vision', 'slow healing wounds', 'numbness in feet', 'weight loss'], description: 'A metabolic disorder where the body cannot properly use insulin, leading to high blood sugar levels.', precautions: ['Monitor blood sugar regularly', 'Follow a low-sugar diet', 'Exercise 30 minutes daily', 'Take medications as prescribed', 'Regular foot and eye checkups'] },
    { name: 'Migraine', symptoms: ['severe headache', 'throbbing pain', 'nausea', 'vomiting', 'sensitivity to light', 'sensitivity to sound', 'aura', 'visual disturbances'], description: 'A neurological condition causing severe throbbing headache, usually on one side of the head.', precautions: ['Identify and avoid triggers', 'Maintain regular sleep schedule', 'Stay hydrated', 'Manage stress', 'Take prescribed preventive medications'] },
    { name: 'Asthma', symptoms: ['wheezing', 'shortness of breath', 'chest tightness', 'coughing', 'difficulty breathing', 'breathlessness at night'], description: 'A chronic respiratory condition with airway inflammation and narrowing causing breathing difficulties.', precautions: ['Avoid allergens and triggers', 'Always carry rescue inhaler', 'Monitor peak flow', 'Avoid smoking', 'Get flu vaccine annually'] },
    { name: 'Dengue Fever', symptoms: ['high fever', 'severe headache', 'pain behind eyes', 'joint pain', 'muscle pain', 'rash', 'mild bleeding', 'fatigue', 'nausea'], description: 'A mosquito-borne viral infection causing flu-like illness. Severe dengue can be life-threatening.', precautions: ['Use mosquito repellent', 'Wear long sleeves', 'Eliminate standing water', 'Stay hydrated', 'Avoid aspirin and ibuprofen — use paracetamol only'] },
    { name: 'Malaria', symptoms: ['cyclical fever', 'chills', 'sweating', 'headache', 'nausea', 'vomiting', 'muscle pain', 'fatigue', 'shivering'], description: 'A parasitic disease transmitted by Anopheles mosquitoes. Plasmodium falciparum is the most dangerous type.', precautions: ['Use mosquito nets', 'Take antimalarial prophylaxis if traveling', 'Use insect repellent', 'Wear protective clothing', 'Seek immediate treatment if symptoms appear'] },
    { name: 'Typhoid', symptoms: ['sustained fever', 'weakness', 'stomach pain', 'headache', 'diarrhea', 'constipation', 'rose spots on skin', 'loss of appetite'], description: 'A bacterial infection caused by Salmonella typhi, spread through contaminated food and water.', precautions: ['Drink only boiled or bottled water', 'Eat freshly cooked food', 'Wash hands frequently', 'Get typhoid vaccine', 'Avoid raw vegetables and fruits in endemic areas'] },
    { name: 'COVID-19', symptoms: ['fever', 'dry cough', 'shortness of breath', 'fatigue', 'loss of taste', 'loss of smell', 'body aches', 'sore throat', 'headache'], description: 'A respiratory illness caused by SARS-CoV-2 coronavirus. Ranges from mild to severe.', precautions: ['Wear mask in crowded places', 'Wash hands frequently', 'Maintain social distance', 'Get vaccinated', 'Isolate if symptomatic'] },
    { name: 'Pneumonia', symptoms: ['cough with phlegm', 'fever', 'chills', 'difficulty breathing', 'chest pain', 'fatigue', 'nausea', 'rapid breathing'], description: 'Lung infection causing inflammation of air sacs, which may fill with fluid or pus.', precautions: ['Get pneumococcal vaccine', 'Quit smoking', 'Wash hands regularly', 'Treat respiratory infections promptly', 'Maintain good nutrition'] },
    { name: 'Appendicitis', symptoms: ['sudden abdominal pain', 'pain around navel', 'pain in lower right abdomen', 'nausea', 'vomiting', 'fever', 'loss of appetite', 'rebound tenderness'], description: 'Inflammation of the appendix. MEDICAL EMERGENCY — can rupture if untreated.', precautions: ['Seek emergency care immediately', 'Do not eat or drink', 'Do not apply heat to abdomen', 'Do not take laxatives', 'Surgery (appendectomy) is the treatment'] },
  ];

  builtin.forEach(d => {
    const exists = knowledgeBase.diseases.find(x => x.nameLower === d.name.toLowerCase());
    if (!exists) {
      knowledgeBase.diseases.push({
        ...d,
        nameLower: d.name.toLowerCase(),
        source: 'builtin',
      });
    }
  });
  console.log(`  Built-in: ${builtin.length} diseases`);
}

// ─── INIT ─────────────────────────────────────────────────────────────────────

let initialized = false;

export async function initKnowledge() {
  if (initialized) return;
  console.log('\n📚 Loading knowledge base into memory...');

  loadBuiltin();
  loadDSP();
  loadSymptom2Disease();
  await loadMedQuAD();

  initialized = true;
  console.log(`\n✅ Knowledge base ready:`);
  console.log(`   Diseases/conditions: ${knowledgeBase.diseases.length}`);
  console.log(`   Q&A pairs: ${knowledgeBase.qa.length}`);
  console.log(`   Symptom severity entries: ${Object.keys(knowledgeBase.symptomSeverity).length}\n`);
}
