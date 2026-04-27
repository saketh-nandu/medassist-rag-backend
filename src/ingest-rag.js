/**
 * ingest-rag.js
 * ACTUAL RAG ingestion pipeline:
 *   1. Load documents from DSP, MedQuAD, Symptom2Disease, BuiltinKnowledge
 *   2. CHUNK each document into overlapping text windows
 *   3. EMBED each chunk using all-MiniLM-L6-v2 (384-dim vectors)
 *   4. STORE chunks + embeddings in Supabase medical_chunks table
 *
 * Usage:
 *   node src/ingest-rag.js            -- full ingest
 *   node src/ingest-rag.js --dry-run  -- preview counts only
 *   node src/ingest-rag.js --limit 100 -- ingest first 100 docs (for testing)
 */

import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import { parseStringPromise } from 'xml2js';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import { chunkDocument } from './chunk.js';
import { embedBatch, warmupEmbedder } from './embed.js';
dotenv.config();

const PROJECT_DIR = path.resolve(process.cwd(), '..', 'project');
const MEDQUAD_DIR = path.join(PROJECT_DIR, 'MedQuAD-master');
const DSP_DIR     = path.join(PROJECT_DIR, 'DSP');
const S2D_FILE    = path.join(PROJECT_DIR, 'Symptom2Disease.csv');

const DRY_RUN   = process.argv.includes('--dry-run');
const LIMIT_ARG = process.argv.indexOf('--limit');
const DOC_LIMIT = LIMIT_ARG !== -1 ? parseInt(process.argv[LIMIT_ARG + 1]) : Infinity;

const EMBED_BATCH = 32;  // embed N chunks at a time
const DB_BATCH    = 50;  // insert N rows at a time

const supabase = DRY_RUN ? null : createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// ─── HELPERS ─────────────────────────────────────────────────────────────────

function clean(t) { return (t || '').replace(/\s+/g, ' ').trim(); }
function cleanSym(s) { return (s || '').replace(/_/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase(); }

// ─── DOCUMENT LOADERS (same as ingest.js) ────────────────────────────────────

function loadBuiltin() {
  const data = [
    { title: 'Hypertension', symptoms: ['headache','dizziness','blurred vision','chest pain','shortness of breath','nosebleed','fatigue'], description: 'High blood pressure (above 130/80 mmHg). Often called the silent killer. Risk factors: obesity, smoking, high salt diet, stress, family history.', precautions: ['Reduce salt intake','Exercise regularly','Avoid smoking and alcohol','Monitor blood pressure daily','Take prescribed medications'] },
    { title: 'Type 2 Diabetes', symptoms: ['increased thirst','frequent urination','fatigue','blurred vision','slow healing wounds','numbness in feet','weight loss','hunger'], description: 'A metabolic disorder where the body cannot properly use insulin, leading to high blood sugar levels.', precautions: ['Monitor blood sugar regularly','Follow a low-sugar diet','Exercise 30 minutes daily','Take medications as prescribed','Regular foot and eye checkups'] },
    { title: 'Migraine', symptoms: ['severe headache','throbbing pain','nausea','vomiting','sensitivity to light','sensitivity to sound','aura','visual disturbances','one sided headache'], description: 'A neurological condition causing severe throbbing headache, usually on one side of the head. Can last 4-72 hours.', precautions: ['Identify and avoid triggers','Maintain regular sleep schedule','Stay hydrated','Manage stress','Take prescribed preventive medications'] },
    { title: 'Asthma', symptoms: ['wheezing','shortness of breath','chest tightness','coughing','difficulty breathing','breathlessness at night','breathlessness during exercise'], description: 'A chronic respiratory condition with airway inflammation and narrowing causing breathing difficulties.', precautions: ['Avoid allergens and triggers','Always carry rescue inhaler','Monitor peak flow','Avoid smoking','Get flu vaccine annually'] },
    { title: 'Dengue Fever', symptoms: ['high fever','severe headache','pain behind eyes','joint pain','muscle pain','rash','mild bleeding','fatigue','nausea','vomiting'], description: 'A mosquito-borne viral infection. Severe dengue can cause plasma leakage and organ damage.', precautions: ['Use mosquito repellent','Wear long sleeves','Eliminate standing water','Stay hydrated','Avoid aspirin and ibuprofen — use paracetamol only'] },
    { title: 'Malaria', symptoms: ['cyclical fever','chills','sweating','headache','nausea','vomiting','muscle pain','fatigue','shivering','high temperature'], description: 'A parasitic disease transmitted by Anopheles mosquitoes.', precautions: ['Use mosquito nets','Take antimalarial prophylaxis if traveling','Use insect repellent','Wear protective clothing','Seek immediate treatment if symptoms appear'] },
    { title: 'Typhoid', symptoms: ['sustained fever','weakness','stomach pain','headache','diarrhea','constipation','loss of appetite','rose spots on skin'], description: 'A bacterial infection caused by Salmonella typhi, spread through contaminated food and water.', precautions: ['Drink only boiled or bottled water','Eat freshly cooked food','Wash hands frequently','Get typhoid vaccine','Avoid raw vegetables in endemic areas'] },
    { title: 'COVID-19', symptoms: ['fever','dry cough','shortness of breath','fatigue','loss of taste','loss of smell','body aches','sore throat','headache','runny nose'], description: 'A respiratory illness caused by SARS-CoV-2 coronavirus. Ranges from mild to severe.', precautions: ['Wear mask in crowded places','Wash hands frequently','Maintain social distance','Get vaccinated','Isolate if symptomatic'] },
    { title: 'Pneumonia', symptoms: ['cough with phlegm','fever','chills','difficulty breathing','chest pain','fatigue','nausea','rapid breathing','sweating'], description: 'Lung infection causing inflammation of air sacs, which may fill with fluid or pus.', precautions: ['Get pneumococcal vaccine','Quit smoking','Wash hands regularly','Treat respiratory infections promptly','Maintain good nutrition'] },
    { title: 'Appendicitis', symptoms: ['sudden abdominal pain','pain around navel','pain in lower right abdomen','nausea','vomiting','fever','loss of appetite','rebound tenderness'], description: 'Inflammation of the appendix. MEDICAL EMERGENCY — can rupture if untreated. Requires immediate surgery.', precautions: ['Seek emergency care immediately','Do not eat or drink','Do not apply heat to abdomen','Do not take laxatives','Surgery (appendectomy) is the treatment'] },
    { title: 'Fracture', symptoms: ['bone pain','swelling','bruising','deformity','inability to move limb','tenderness','crepitus'], description: 'A break in bone continuity. Can be closed (skin intact) or open (bone pierces skin).', precautions: ['Immobilize the injured area','Apply ice wrapped in cloth','Elevate the limb','Seek emergency care','Do not attempt to straighten the bone'] },
    { title: 'Gastroenteritis', symptoms: ['diarrhea','vomiting','nausea','stomach cramps','fever','dehydration','loss of appetite','muscle aches'], description: 'Inflammation of the stomach and intestines, usually caused by viral or bacterial infection.', precautions: ['Stay hydrated with ORS','Eat bland foods (BRAT diet)','Wash hands frequently','Avoid dairy and fatty foods','Seek care if dehydration is severe'] },
    { title: 'Urinary Tract Infection', symptoms: ['burning urination','frequent urination','cloudy urine','strong smelling urine','pelvic pain','blood in urine','lower abdominal pain'], description: 'Bacterial infection of the urinary system. More common in women.', precautions: ['Drink plenty of water','Urinate after intercourse','Wipe front to back','Avoid holding urine','Complete full course of antibiotics'] },
    { title: 'Anemia', symptoms: ['fatigue','weakness','pale skin','shortness of breath','dizziness','cold hands and feet','headache','chest pain','irregular heartbeat'], description: 'A condition where you lack enough healthy red blood cells to carry adequate oxygen to body tissues.', precautions: ['Eat iron-rich foods','Take vitamin C with iron foods','Take prescribed iron supplements','Treat underlying cause','Regular blood tests'] },
    { title: 'Chickenpox', symptoms: ['itchy rash','blisters','fever','fatigue','loss of appetite','headache','fluid-filled spots'], description: 'A highly contagious viral infection caused by varicella-zoster virus. Characterized by itchy blister-like rash.', precautions: ['Get vaccinated','Avoid scratching','Use calamine lotion','Take antihistamines for itching','Isolate until all blisters crust over'] },
  ];
  return data.map(d => ({
    source: 'BuiltinKnowledge', category: 'disease', title: d.title,
    content: `Disease: ${d.title}\nDescription: ${d.description}\nSymptoms: ${d.symptoms.join(', ')}\nPrecautions: ${d.precautions.join('. ')}`,
    symptoms: d.symptoms, precautions: d.precautions, metadata: { builtin: true },
  }));
}

function loadDSP() {
  const rows = [];
  const diseaseMap = {};
  const datasetFile = path.join(DSP_DIR, 'dataset.csv');
  if (fs.existsSync(datasetFile)) {
    const data = parse(fs.readFileSync(datasetFile, 'utf8'), { columns: true, skip_empty_lines: true, trim: true });
    data.forEach(row => {
      const name = clean(row.Disease || '');
      if (!name) return;
      if (!diseaseMap[name]) diseaseMap[name] = { symptoms: new Set(), description: '', precautions: [] };
      Object.entries(row).forEach(([col, val]) => {
        if (col === 'Disease') return;
        const s = cleanSym(val);
        if (s) diseaseMap[name].symptoms.add(s);
      });
    });
  }
  const descFile = path.join(DSP_DIR, 'symptom_Description.csv');
  if (fs.existsSync(descFile)) {
    const data = parse(fs.readFileSync(descFile, 'utf8'), { columns: true, skip_empty_lines: true, trim: true });
    data.forEach(row => {
      const name = clean(row.Disease || '');
      if (name && diseaseMap[name]) diseaseMap[name].description = clean(row.Description || '');
    });
  }
  const precFile = path.join(DSP_DIR, 'symptom_precaution.csv');
  if (fs.existsSync(precFile)) {
    const data = parse(fs.readFileSync(precFile, 'utf8'), { columns: true, skip_empty_lines: true, trim: true });
    data.forEach(row => {
      const name = clean(row.Disease || '');
      if (!name || !diseaseMap[name]) return;
      diseaseMap[name].precautions = [row.Precaution_1, row.Precaution_2, row.Precaution_3, row.Precaution_4].map(p => clean(p)).filter(Boolean);
    });
  }
  Object.entries(diseaseMap).forEach(([name, d]) => {
    const symptoms = [...d.symptoms].filter(Boolean);
    rows.push({
      source: 'DSP', category: 'disease', title: name,
      content: [`Disease: ${name}`, d.description ? `Description: ${d.description}` : '', symptoms.length > 0 ? `Symptoms: ${symptoms.join(', ')}` : '', d.precautions.length > 0 ? `Precautions: ${d.precautions.join(', ')}` : ''].filter(Boolean).join('\n'),
      symptoms, precautions: d.precautions, metadata: {},
    });
  });
  return rows;
}

function loadSymptom2Disease() {
  if (!fs.existsSync(S2D_FILE)) return [];
  const data = parse(fs.readFileSync(S2D_FILE, 'utf8'), { columns: true, skip_empty_lines: true, trim: true });
  const diseaseMap = {};
  data.forEach(row => {
    const label = clean(row.label || '');
    const text  = clean(row.text || '');
    if (!label || !text) return;
    if (!diseaseMap[label]) diseaseMap[label] = [];
    if (diseaseMap[label].length < 5) diseaseMap[label].push(text);
  });
  return Object.entries(diseaseMap).map(([name, texts]) => ({
    source: 'Symptom2Disease', category: 'disease', title: name,
    content: `Disease: ${name}\nPatient descriptions:\n${texts.map((t, i) => `${i + 1}. ${t}`).join('\n')}`,
    symptoms: [], precautions: [], metadata: {},
  }));
}

async function loadMedQuAD(limit = Infinity) {
  if (!fs.existsSync(MEDQUAD_DIR)) return [];
  const subfolders = fs.readdirSync(MEDQUAD_DIR).filter(d => fs.statSync(path.join(MEDQUAD_DIR, d)).isDirectory());
  const rows = [];
  let fileCount = 0;
  outer: for (const folder of subfolders) {
    const folderPath = path.join(MEDQUAD_DIR, folder);
    const xmlFiles = fs.readdirSync(folderPath).filter(f => f.endsWith('.xml'));
    for (const xmlFile of xmlFiles) {
      if (rows.length >= limit) break outer;
      try {
        const xml = fs.readFileSync(path.join(folderPath, xmlFile), 'utf8');
        const doc = await parseStringPromise(xml, { explicitArray: false });
        const focus = clean(doc?.Document?.Focus || '');
        const qaPairs = doc?.Document?.QAPairs?.QAPair;
        if (!qaPairs || !focus) continue;
        const pairs = Array.isArray(qaPairs) ? qaPairs : [qaPairs];
        const qaTexts = pairs.map(p => {
          const q = clean(p?.Question?._ || p?.Question || '');
          const a = clean(p?.Answer || '');
          return (q && a && a.length > 30) ? `Q: ${q}\nA: ${a.slice(0, 800)}` : null;
        }).filter(Boolean);
        if (qaTexts.length > 0) {
          rows.push({ source: 'MedQuAD', category: 'qa', title: focus, content: `Topic: ${focus}\n\n${qaTexts.join('\n\n')}`, symptoms: [], precautions: [], metadata: { folder, file: xmlFile } });
        }
        fileCount++;
      } catch (_) {}
    }
  }
  console.log(`  MedQuAD → ${fileCount} files → ${rows.length} docs`);
  return rows;
}

// ─── EMBED + STORE ────────────────────────────────────────────────────────────

async function processAndStore(docs) {
  // 1. Chunk all documents
  console.log('\n✂️  Chunking documents...');
  const allChunks = [];
  for (const doc of docs) {
    const chunks = chunkDocument(doc);
    allChunks.push(...chunks);
  }
  console.log(`   ${docs.length} docs → ${allChunks.length} chunks`);

  if (DRY_RUN) {
    console.log('\n✅ Dry run — no embeddings or DB writes.');
    const avgChunks = (allChunks.length / docs.length).toFixed(1);
    console.log(`   Avg chunks per doc: ${avgChunks}`);
    console.log(`   Sample chunk (first):\n   "${allChunks[0]?.content?.slice(0, 200)}..."`);
    return;
  }

  // 2. Clear existing chunks
  console.log('\n🗑️  Clearing existing chunks...');
  const { error: delErr } = await supabase.from('medical_chunks').delete().neq('id', 0);
  if (delErr) console.warn('  Warning clearing table:', delErr.message);

  // 3. Embed in batches and insert
  console.log('\n🔢 Embedding + storing chunks...');
  let stored = 0;
  const total = allChunks.length;

  for (let i = 0; i < allChunks.length; i += EMBED_BATCH) {
    const batch = allChunks.slice(i, i + EMBED_BATCH);
    const texts = batch.map(c => c.content);

    // Generate embeddings
    const embeddings = await embedBatch(texts);

    // Attach embeddings to chunks
    const rows = batch.map((chunk, j) => ({
      ...chunk,
      embedding: `[${embeddings[j].join(',')}]`, // pgvector format
    }));

    // Insert into Supabase in sub-batches
    for (let j = 0; j < rows.length; j += DB_BATCH) {
      const subBatch = rows.slice(j, j + DB_BATCH);
      const { error } = await supabase.from('medical_chunks').insert(subBatch);
      if (error) console.error(`  ❌ Insert error: ${error.message}`);
      else stored += subBatch.length;
    }

    const pct = Math.round(((i + batch.length) / total) * 100);
    process.stdout.write(`\r  Progress: ${pct}% (${i + batch.length}/${total} chunks embedded)`);
  }

  console.log(`\n\n✅ Stored ${stored}/${total} chunks with embeddings.`);
}

// ─── MAIN ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log('\n🚀 HosFind REAL RAG Ingestion Pipeline');
  console.log(`   Mode: ${DRY_RUN ? 'DRY RUN' : 'LIVE → Supabase'}`);
  if (DOC_LIMIT !== Infinity) console.log(`   Limit: ${DOC_LIMIT} docs per source`);
  console.log();

  // Warm up embedder
  if (!DRY_RUN) {
    console.log('🔥 Warming up embedding model...');
    await warmupEmbedder();
  }

  const allDocs = [];

  console.log('📚 Loading BuiltinKnowledge...');
  const builtin = loadBuiltin();
  console.log(`   ✅ ${builtin.length} docs`);
  allDocs.push(...builtin);

  console.log('📚 Loading DSP dataset...');
  const dsp = loadDSP();
  console.log(`   ✅ ${dsp.length} docs`);
  allDocs.push(...dsp);

  console.log('📚 Loading Symptom2Disease...');
  const s2d = loadSymptom2Disease();
  console.log(`   ✅ ${s2d.length} docs`);
  allDocs.push(...s2d);

  console.log('📚 Loading MedQuAD XML...');
  const medquad = await loadMedQuAD(DOC_LIMIT);
  console.log(`   ✅ ${medquad.length} docs`);
  allDocs.push(...medquad);

  console.log(`\n📊 Total documents: ${allDocs.length}`);

  await processAndStore(allDocs);

  const bySource = {};
  allDocs.forEach(d => { bySource[d.source] = (bySource[d.source] || 0) + 1; });
  console.log('\nBreakdown by source:', bySource);
}

main().catch(err => {
  console.error('\n❌ Fatal:', err.message);
  process.exit(1);
});
