/**
 * ingest.js
 * Parses all datasets and inserts into Supabase medical_knowledge table.
 * No embeddings — just structured text for full-text search.
 *
 * Usage:
 *   node src/ingest.js            -- ingest everything
 *   node src/ingest.js --dry-run  -- preview counts only
 */

import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import { parseStringPromise } from 'xml2js';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
dotenv.config();

const PROJECT_DIR = path.resolve(process.cwd(), '..', 'project');
const MEDQUAD_DIR = path.join(PROJECT_DIR, 'MedQuAD-master');
const DSP_DIR     = path.join(PROJECT_DIR, 'DSP');
const S2D_FILE    = path.join(PROJECT_DIR, 'Symptom2Disease.csv');

const DRY_RUN      = process.argv.includes('--dry-run');
const BATCH_SIZE   = 100;

const supabase = DRY_RUN ? null : createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// ─── HELPERS ─────────────────────────────────────────────────────────────────

function clean(t) {
  return (t || '').replace(/\s+/g, ' ').trim();
}

function cleanSym(s) {
  return (s || '').replace(/_/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
}

// ─── 1. DSP DATASET ──────────────────────────────────────────────────────────

function loadDSP() {
  const rows = [];
  const diseaseMap = {}; // name → { symptoms, description, precautions }

  // dataset.csv — disease + symptom columns
  const datasetFile = path.join(DSP_DIR, 'dataset.csv');
  if (fs.existsSync(datasetFile)) {
    const data = parse(fs.readFileSync(datasetFile, 'utf8'), {
      columns: true, skip_empty_lines: true, trim: true,
    });
    data.forEach(row => {
      const name = clean(row.Disease || '');
      if (!name) return;
      if (!diseaseMap[name]) diseaseMap[name] = { symptoms: new Set(), description: '', precautions: [] };
      // All other columns are symptoms
      Object.entries(row).forEach(([col, val]) => {
        if (col === 'Disease') return;
        const s = cleanSym(val);
        if (s) diseaseMap[name].symptoms.add(s);
      });
    });
    console.log(`  dataset.csv → ${Object.keys(diseaseMap).length} diseases`);
  }

  // symptom_Description.csv
  const descFile = path.join(DSP_DIR, 'symptom_Description.csv');
  if (fs.existsSync(descFile)) {
    const data = parse(fs.readFileSync(descFile, 'utf8'), {
      columns: true, skip_empty_lines: true, trim: true,
    });
    data.forEach(row => {
      const name = clean(row.Disease || '');
      const desc = clean(row.Description || '');
      if (!name || !desc) return;
      if (!diseaseMap[name]) diseaseMap[name] = { symptoms: new Set(), description: '', precautions: [] };
      diseaseMap[name].description = desc;
    });
    console.log(`  symptom_Description.csv → ${data.length} descriptions`);
  }

  // symptom_precaution.csv
  const precFile = path.join(DSP_DIR, 'symptom_precaution.csv');
  if (fs.existsSync(precFile)) {
    const data = parse(fs.readFileSync(precFile, 'utf8'), {
      columns: true, skip_empty_lines: true, trim: true,
    });
    data.forEach(row => {
      const name = clean(row.Disease || '');
      const precs = [row.Precaution_1, row.Precaution_2, row.Precaution_3, row.Precaution_4]
        .map(p => clean(p)).filter(Boolean);
      if (!name) return;
      if (!diseaseMap[name]) diseaseMap[name] = { symptoms: new Set(), description: '', precautions: [] };
      diseaseMap[name].precautions = precs;
    });
    console.log(`  symptom_precaution.csv → ${data.length} precautions`);
  }

  // Build rows
  Object.entries(diseaseMap).forEach(([name, d]) => {
    const symptoms = [...d.symptoms].filter(Boolean);
    const content = [
      `Disease: ${name}`,
      d.description ? `Description: ${d.description}` : '',
      symptoms.length > 0 ? `Symptoms: ${symptoms.join(', ')}` : '',
      d.precautions.length > 0 ? `Precautions: ${d.precautions.join(', ')}` : '',
    ].filter(Boolean).join('\n');

    rows.push({
      source:     'DSP',
      category:   'disease',
      title:      name,
      content,
      symptoms,
      precautions: d.precautions,
      metadata:   { has_description: !!d.description },
    });
  });

  return rows;
}

// ─── 2. SYMPTOM2DISEASE ───────────────────────────────────────────────────────

function loadSymptom2Disease() {
  if (!fs.existsSync(S2D_FILE)) {
    console.log('  ⚠️  Symptom2Disease.csv not found');
    return [];
  }

  const data = parse(fs.readFileSync(S2D_FILE, 'utf8'), {
    columns: true, skip_empty_lines: true, trim: true,
  });

  // Group by disease, take up to 3 representative descriptions
  const diseaseMap = {};
  data.forEach(row => {
    const label = clean(row.label || '');
    const text  = clean(row.text || '');
    if (!label || !text) return;
    if (!diseaseMap[label]) diseaseMap[label] = [];
    if (diseaseMap[label].length < 3) diseaseMap[label].push(text);
  });

  const rows = Object.entries(diseaseMap).map(([name, texts]) => ({
    source:     'Symptom2Disease',
    category:   'disease',
    title:      name,
    content:    `Disease: ${name}\nPatient descriptions:\n${texts.map((t, i) => `${i + 1}. ${t}`).join('\n')}`,
    symptoms:   [],
    precautions: [],
    metadata:   { sample_count: texts.length },
  }));

  console.log(`  Symptom2Disease.csv → ${data.length} rows → ${rows.length} diseases`);
  return rows;
}

// ─── 3. MEDQUAD XML ──────────────────────────────────────────────────────────

async function loadMedQuAD() {
  if (!fs.existsSync(MEDQUAD_DIR)) {
    console.log('  ⚠️  MedQuAD-master not found');
    return [];
  }

  const subfolders = fs.readdirSync(MEDQUAD_DIR)
    .filter(d => fs.statSync(path.join(MEDQUAD_DIR, d)).isDirectory());

  const rows = [];
  let fileCount = 0;

  for (const folder of subfolders) {
    const folderPath = path.join(MEDQUAD_DIR, folder);
    const xmlFiles = fs.readdirSync(folderPath).filter(f => f.endsWith('.xml'));

    for (const xmlFile of xmlFiles) {
      try {
        const xml = fs.readFileSync(path.join(folderPath, xmlFile), 'utf8');
        const doc = await parseStringPromise(xml, { explicitArray: false });
        const focus = clean(doc?.Document?.Focus || '');
        const qaPairs = doc?.Document?.QAPairs?.QAPair;
        if (!qaPairs) continue;

        const pairs = Array.isArray(qaPairs) ? qaPairs : [qaPairs];

        // Group all QA for this document into one row per focus topic
        const qaTexts = [];
        pairs.forEach(pair => {
          const q = clean(pair?.Question?._ || pair?.Question || '');
          const a = clean(pair?.Answer || '');
          if (q && a && a.length > 30) {
            qaTexts.push(`Q: ${q}\nA: ${a.slice(0, 600)}`);
          }
        });

        if (qaTexts.length > 0 && focus) {
          // Split into chunks of 3 QA pairs per row to keep content manageable
          for (let i = 0; i < qaTexts.length; i += 3) {
            const chunk = qaTexts.slice(i, i + 3);
            rows.push({
              source:     'MedQuAD',
              category:   'qa',
              title:      focus,
              content:    `Topic: ${focus}\n\n${chunk.join('\n\n')}`,
              symptoms:   [],
              precautions: [],
              metadata:   { folder, file: xmlFile, chunk_index: Math.floor(i / 3) },
            });
          }
        }
        fileCount++;
      } catch (_) {}
    }
  }

  console.log(`  MedQuAD → ${fileCount} files → ${rows.length} rows`);
  return rows;
}

// ─── 4. BUILT-IN ─────────────────────────────────────────────────────────────

function loadBuiltin() {
  const data = [
    { title: 'Hypertension', symptoms: ['headache', 'dizziness', 'blurred vision', 'chest pain', 'shortness of breath', 'nosebleed', 'fatigue'], description: 'High blood pressure (above 130/80 mmHg). Often called the silent killer. Risk factors: obesity, smoking, high salt diet, stress, family history.', precautions: ['Reduce salt intake', 'Exercise regularly', 'Avoid smoking and alcohol', 'Monitor blood pressure daily', 'Take prescribed medications'] },
    { title: 'Type 2 Diabetes', symptoms: ['increased thirst', 'frequent urination', 'fatigue', 'blurred vision', 'slow healing wounds', 'numbness in feet', 'weight loss', 'hunger'], description: 'A metabolic disorder where the body cannot properly use insulin, leading to high blood sugar levels.', precautions: ['Monitor blood sugar regularly', 'Follow a low-sugar diet', 'Exercise 30 minutes daily', 'Take medications as prescribed', 'Regular foot and eye checkups'] },
    { title: 'Migraine', symptoms: ['severe headache', 'throbbing pain', 'nausea', 'vomiting', 'sensitivity to light', 'sensitivity to sound', 'aura', 'visual disturbances', 'one sided headache'], description: 'A neurological condition causing severe throbbing headache, usually on one side of the head. Can last 4-72 hours.', precautions: ['Identify and avoid triggers', 'Maintain regular sleep schedule', 'Stay hydrated', 'Manage stress', 'Take prescribed preventive medications'] },
    { title: 'Asthma', symptoms: ['wheezing', 'shortness of breath', 'chest tightness', 'coughing', 'difficulty breathing', 'breathlessness at night', 'breathlessness during exercise'], description: 'A chronic respiratory condition with airway inflammation and narrowing causing breathing difficulties.', precautions: ['Avoid allergens and triggers', 'Always carry rescue inhaler', 'Monitor peak flow', 'Avoid smoking', 'Get flu vaccine annually'] },
    { title: 'Dengue Fever', symptoms: ['high fever', 'severe headache', 'pain behind eyes', 'joint pain', 'muscle pain', 'rash', 'mild bleeding', 'fatigue', 'nausea', 'vomiting'], description: 'A mosquito-borne viral infection. Severe dengue can cause plasma leakage and organ damage.', precautions: ['Use mosquito repellent', 'Wear long sleeves', 'Eliminate standing water', 'Stay hydrated', 'Avoid aspirin and ibuprofen — use paracetamol only'] },
    { title: 'Malaria', symptoms: ['cyclical fever', 'chills', 'sweating', 'headache', 'nausea', 'vomiting', 'muscle pain', 'fatigue', 'shivering', 'high temperature'], description: 'A parasitic disease transmitted by Anopheles mosquitoes. Plasmodium falciparum is the most dangerous type.', precautions: ['Use mosquito nets', 'Take antimalarial prophylaxis if traveling', 'Use insect repellent', 'Wear protective clothing', 'Seek immediate treatment if symptoms appear'] },
    { title: 'Typhoid', symptoms: ['sustained fever', 'weakness', 'stomach pain', 'headache', 'diarrhea', 'constipation', 'loss of appetite', 'rose spots on skin'], description: 'A bacterial infection caused by Salmonella typhi, spread through contaminated food and water.', precautions: ['Drink only boiled or bottled water', 'Eat freshly cooked food', 'Wash hands frequently', 'Get typhoid vaccine', 'Avoid raw vegetables in endemic areas'] },
    { title: 'COVID-19', symptoms: ['fever', 'dry cough', 'shortness of breath', 'fatigue', 'loss of taste', 'loss of smell', 'body aches', 'sore throat', 'headache', 'runny nose'], description: 'A respiratory illness caused by SARS-CoV-2 coronavirus. Ranges from mild to severe.', precautions: ['Wear mask in crowded places', 'Wash hands frequently', 'Maintain social distance', 'Get vaccinated', 'Isolate if symptomatic'] },
    { title: 'Pneumonia', symptoms: ['cough with phlegm', 'fever', 'chills', 'difficulty breathing', 'chest pain', 'fatigue', 'nausea', 'rapid breathing', 'sweating'], description: 'Lung infection causing inflammation of air sacs, which may fill with fluid or pus.', precautions: ['Get pneumococcal vaccine', 'Quit smoking', 'Wash hands regularly', 'Treat respiratory infections promptly', 'Maintain good nutrition'] },
    { title: 'Appendicitis', symptoms: ['sudden abdominal pain', 'pain around navel', 'pain in lower right abdomen', 'nausea', 'vomiting', 'fever', 'loss of appetite', 'rebound tenderness'], description: 'Inflammation of the appendix. MEDICAL EMERGENCY — can rupture if untreated. Requires immediate surgery.', precautions: ['Seek emergency care immediately', 'Do not eat or drink', 'Do not apply heat to abdomen', 'Do not take laxatives', 'Surgery (appendectomy) is the treatment'] },
    { title: 'Fracture', symptoms: ['bone pain', 'swelling', 'bruising', 'deformity', 'inability to move limb', 'tenderness', 'crepitus'], description: 'A break in bone continuity. Can be closed (skin intact) or open (bone pierces skin).', precautions: ['Immobilize the injured area', 'Apply ice wrapped in cloth', 'Elevate the limb', 'Seek emergency care', 'Do not attempt to straighten the bone'] },
    { title: 'Sprain', symptoms: ['joint pain', 'swelling', 'bruising', 'limited range of motion', 'instability', 'tenderness around joint'], description: 'Stretching or tearing of ligaments. Most common in ankle, wrist, and knee.', precautions: ['Rest the injured area', 'Apply ice for 20 minutes every 2 hours', 'Use compression bandage', 'Elevate the limb', 'Take NSAIDs for pain relief'] },
    { title: 'Gastroenteritis', symptoms: ['diarrhea', 'vomiting', 'nausea', 'stomach cramps', 'fever', 'dehydration', 'loss of appetite', 'muscle aches'], description: 'Inflammation of the stomach and intestines, usually caused by viral or bacterial infection.', precautions: ['Stay hydrated with ORS', 'Eat bland foods (BRAT diet)', 'Wash hands frequently', 'Avoid dairy and fatty foods', 'Seek care if dehydration is severe'] },
    { title: 'Urinary Tract Infection', symptoms: ['burning urination', 'frequent urination', 'cloudy urine', 'strong smelling urine', 'pelvic pain', 'blood in urine', 'lower abdominal pain'], description: 'Bacterial infection of the urinary system. More common in women.', precautions: ['Drink plenty of water', 'Urinate after intercourse', 'Wipe front to back', 'Avoid holding urine', 'Complete full course of antibiotics'] },
    { title: 'Anemia', symptoms: ['fatigue', 'weakness', 'pale skin', 'shortness of breath', 'dizziness', 'cold hands and feet', 'headache', 'chest pain', 'irregular heartbeat'], description: 'A condition where you lack enough healthy red blood cells to carry adequate oxygen to body tissues.', precautions: ['Eat iron-rich foods', 'Take vitamin C with iron foods', 'Take prescribed iron supplements', 'Treat underlying cause', 'Regular blood tests'] },
  ];

  return data.map(d => ({
    source:     'BuiltinKnowledge',
    category:   'disease',
    title:      d.title,
    content:    `Disease: ${d.title}\nDescription: ${d.description}\nSymptoms: ${d.symptoms.join(', ')}\nPrecautions: ${d.precautions.join(', ')}`,
    symptoms:   d.symptoms,
    precautions: d.precautions,
    metadata:   { builtin: true },
  }));
}

// ─── UPSERT TO SUPABASE ───────────────────────────────────────────────────────

async function upsertBatch(rows) {
  let total = 0;
  for (let i = 0; i < rows.length; i += BATCH_SIZE) {
    const batch = rows.slice(i, i + BATCH_SIZE);
    const { error } = await supabase.from('medical_knowledge').insert(batch);
    if (error) {
      console.error(`  ❌ Batch ${i}: ${error.message}`);
    } else {
      total += batch.length;
    }
    process.stdout.write(`\r  Inserted ${total}/${rows.length}`);
  }
  console.log();
  return total;
}

// ─── MAIN ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log('\n🚀 HosFind Ingestion Pipeline');
  console.log(`Mode: ${DRY_RUN ? 'DRY RUN' : 'LIVE → Supabase'}\n`);

  const allRows = [];

  console.log('📚 Built-in knowledge...');
  const builtin = loadBuiltin();
  console.log(`  ✅ ${builtin.length} rows`);
  allRows.push(...builtin);

  console.log('\n📚 DSP dataset...');
  const dsp = loadDSP();
  console.log(`  ✅ ${dsp.length} rows`);
  allRows.push(...dsp);

  console.log('\n📚 Symptom2Disease...');
  const s2d = loadSymptom2Disease();
  console.log(`  ✅ ${s2d.length} rows`);
  allRows.push(...s2d);

  console.log('\n📚 MedQuAD XML...');
  const medquad = await loadMedQuAD();
  console.log(`  ✅ ${medquad.length} rows`);
  allRows.push(...medquad);

  console.log(`\n📊 Total rows: ${allRows.length}`);

  if (DRY_RUN) {
    console.log('\n✅ Dry run complete — no data written.');
    const bySource = {};
    allRows.forEach(r => { bySource[r.source] = (bySource[r.source] || 0) + 1; });
    console.log('Breakdown:', bySource);
    return;
  }

  console.log('\n💾 Inserting into Supabase...');
  const inserted = await upsertBatch(allRows);
  console.log(`\n✅ Done! ${inserted} rows in Supabase.`);

  const bySource = {};
  allRows.forEach(r => { bySource[r.source] = (bySource[r.source] || 0) + 1; });
  console.log('Breakdown:', bySource);
}

main().catch(err => {
  console.error('Fatal:', err.message);
  process.exit(1);
});
