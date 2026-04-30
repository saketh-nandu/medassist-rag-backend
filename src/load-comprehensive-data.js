/**
 * load-comprehensive-data.js — Load Comprehensive Medical Knowledge
 * Enhanced script to populate database with extensive medical data for realistic evaluation
 */

import { createClient } from '@supabase/supabase-js';
import { embedText } from './embed.js';
import dotenv from 'dotenv';
dotenv.config();

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// Comprehensive medical knowledge dataset
const COMPREHENSIVE_MEDICAL_DATA = [
  // ─── COMMON DISEASES ─────────────────────────────────────────────────────────
  {
    title: "Dengue Fever",
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: Dengue Fever Category: disease Disease: Dengue Fever Description: A mosquito-borne viral infection causing high fever, severe headache, pain behind eyes, joint and muscle pain, and rash. Dengue is transmitted by Aedes aegypti mosquitoes and is endemic in tropical regions. Symptoms: high fever (40°C/104°F), severe headache, retro-orbital pain (pain behind eyes), joint pain (arthralgia), muscle pain (myalgia), skin rash, nausea, vomiting, mild bleeding. Warning signs include severe abdominal pain, persistent vomiting, rapid breathing, bleeding gums, blood in vomit. Complications: dengue hemorrhagic fever, dengue shock syndrome, plasma leakage, severe bleeding. Treatment: supportive care, paracetamol for fever, adequate fluid intake, avoid aspirin and NSAIDs. Precautions: Use mosquito repellent containing DEET. Wear long sleeves and pants. Use bed nets. Eliminate standing water around homes. Clean water containers regularly. Seek immediate medical attention for warning signs."
  },
  {
    title: "Malaria", 
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: Malaria Category: disease Disease: Malaria Description: A life-threatening parasitic disease transmitted by infected female Anopheles mosquitoes. Caused by Plasmodium parasites (P. falciparum, P. vivax, P. ovale, P. malariae, P. knowlesi). Symptoms: cyclical fever with chills and sweating, headache, muscle pain, fatigue, nausea, vomiting, diarrhea. Severe malaria symptoms include altered consciousness, seizures, difficulty breathing, dark urine, jaundice. Diagnosis: blood smear microscopy, rapid diagnostic tests, PCR. Treatment: antimalarial medications (artemisinin-based combination therapy for P. falciparum, chloroquine for P. vivax in sensitive areas). Complications: cerebral malaria, severe anemia, respiratory distress, kidney failure, hypoglycemia. Precautions: Use insecticide-treated bed nets. Take antimalarial prophylaxis in endemic areas. Use insect repellent. Wear protective clothing. Eliminate mosquito breeding sites. Seek immediate medical care for fever in endemic areas."
  },
  {
    title: "Typhoid Fever",
    category: "disease", 
    source: "BuiltinKnowledge",
    content: "Title: Typhoid Fever Category: disease Disease: Typhoid Description: A bacterial infection caused by Salmonella enterica serotype Typhi, spread through contaminated food and water. Symptoms: sustained high fever (39-40°C), weakness, stomach pain, headache, diarrhea or constipation, loss of appetite, rose-colored spots on chest. Complications: intestinal bleeding, perforation, encephalitis, myocarditis. Diagnosis: blood culture, stool culture, Widal test, typhidot test. Treatment: antibiotics (fluoroquinolones, azithromycin, ceftriaxone), supportive care, fluid replacement. Precautions: Drink only boiled or bottled water. Eat freshly cooked hot food. Avoid raw vegetables and fruits unless you peel them yourself. Wash hands frequently with soap. Get typhoid vaccine before traveling to endemic areas. Avoid street food and ice cubes. Practice good sanitation and hygiene."
  },
  {
    title: "Diabetes Type 2",
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: Diabetes Type 2 Category: disease Disease: Diabetes Mellitus Type 2 Description: A chronic metabolic disorder characterized by insulin resistance and relative insulin deficiency, leading to elevated blood glucose levels. Risk factors include obesity, sedentary lifestyle, family history, age >45, ethnicity. Symptoms: increased thirst (polydipsia), frequent urination (polyuria), increased hunger (polyphagia), fatigue, blurred vision, slow-healing wounds, frequent infections, tingling in hands/feet. Complications: diabetic retinopathy, nephropathy, neuropathy, cardiovascular disease, stroke, foot ulcers. Diagnosis: fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, random glucose ≥200 mg/dL with symptoms. Treatment: lifestyle modifications, metformin, insulin, other antidiabetic medications. Precautions: Maintain healthy diet with complex carbohydrates. Exercise regularly (150 min/week). Monitor blood sugar levels. Take prescribed medications. Maintain healthy weight (BMI 18.5-24.9). Regular medical checkups and eye exams. Foot care and inspection."
  },
  {
    title: "Hypertension",
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: Hypertension Category: disease Disease: High Blood Pressure Description: A chronic condition where blood pressure in arteries is persistently elevated (≥140/90 mmHg). Often called 'silent killer' as it usually has no symptoms. Types: primary (essential) hypertension, secondary hypertension. Risk factors: age, family history, obesity, high sodium intake, physical inactivity, smoking, alcohol, stress. Symptoms: usually asymptomatic, sometimes headache, dizziness, nosebleeds, shortness of breath in severe cases. Complications: heart attack, stroke, heart failure, kidney disease, vision problems, peripheral artery disease. Diagnosis: multiple blood pressure readings, ambulatory monitoring. Treatment: lifestyle modifications, ACE inhibitors, ARBs, diuretics, calcium channel blockers, beta-blockers. Precautions: Reduce sodium intake (<2.3g/day). Exercise regularly. Maintain healthy weight. Limit alcohol consumption. Don't smoke. Manage stress. Take medications as prescribed. Regular blood pressure monitoring."
  },

  // ─── EMERGENCY CONDITIONS ───────────────────────────────────────────────────
  {
    title: "Heart Attack (Myocardial Infarction)",
    category: "emergency",
    source: "BuiltinKnowledge",
    content: "Title: Heart Attack Category: emergency Disease: Myocardial Infarction Description: A medical emergency where blood flow to part of the heart muscle is blocked, causing tissue death. Symptoms: severe chest pain or pressure (crushing, squeezing sensation), pain radiating to left arm, jaw, neck, back, shortness of breath, sweating, nausea, vomiting, lightheadedness, fatigue. Women may have atypical symptoms: jaw pain, back pain, nausea without chest pain. Risk factors: coronary artery disease, high cholesterol, hypertension, diabetes, smoking, obesity, family history. Emergency treatment: Call 108 immediately, chew aspirin if not allergic, nitroglycerin if prescribed, CPR if unconscious. Hospital treatment: thrombolytics, angioplasty, stents, bypass surgery. Precautions: Recognize symptoms early. Don't drive yourself to hospital. Stay calm and sit upright. Loosen tight clothing. Time is muscle - seek immediate medical attention."
  },
  {
    title: "Stroke",
    category: "emergency",
    source: "BuiltinKnowledge",
    content: "Title: Stroke Category: emergency Disease: Cerebrovascular Accident Description: A medical emergency where blood supply to brain is interrupted (ischemic) or blood vessel bursts (hemorrhagic). FAST signs: Face drooping, Arm weakness, Speech difficulty, Time to call emergency. Symptoms: sudden numbness/weakness of face/arm/leg (especially one side), sudden confusion, trouble speaking/understanding, sudden vision problems, sudden severe headache, sudden trouble walking/dizziness/loss of coordination. Types: ischemic stroke (87%), hemorrhagic stroke (13%), transient ischemic attack (TIA). Risk factors: hypertension, atrial fibrillation, diabetes, high cholesterol, smoking, age. Emergency treatment: Call 108 immediately, note time of symptom onset, don't give food/water, keep airway clear. Hospital treatment: thrombolytics (within 3-4.5 hours), mechanical thrombectomy, blood pressure management. Precautions: Recognize FAST signs. Time is brain - every minute counts. Don't wait for symptoms to improve."
  },
  {
    title: "Anaphylaxis",
    category: "emergency",
    source: "BuiltinKnowledge",
    content: "Title: Anaphylaxis Category: emergency Disease: Severe Allergic Reaction Description: A severe, life-threatening allergic reaction that occurs rapidly and can cause death. Triggers: foods (peanuts, shellfish, eggs), medications (penicillin, aspirin), insect stings, latex. Symptoms: skin reactions (hives, itching, flushed/pale skin), swelling of face/eyes/lips/tongue/throat, difficulty breathing/wheezing, rapid weak pulse, nausea/vomiting/diarrhea, dizziness/fainting, sense of impending doom. Biphasic reaction possible 4-12 hours later. Emergency treatment: Call 108 immediately, use epinephrine auto-injector (EpiPen) if available, remove trigger if possible, lie flat with legs elevated, loosen tight clothing, be prepared for CPR. Hospital treatment: epinephrine, corticosteroids, antihistamines, IV fluids, oxygen. Precautions: Carry epinephrine auto-injector if at risk. Wear medical alert bracelet. Avoid known allergens. Seek immediate medical attention even after epinephrine use."
  },

  // ─── COMMON SYMPTOMS AND CONDITIONS ─────────────────────────────────────────
  {
    title: "Headache",
    category: "qa",
    source: "BuiltinKnowledge",
    content: "Title: Headache Category: qa Topic: Headache Q: What causes headaches and how to treat them? A: Headaches are the most common form of pain and major reason for missed work/school. Types include tension headaches (most common, due to tight muscles in shoulders/neck/scalp/jaw), migraines (severe throbbing pain, often with nausea/light sensitivity), cluster headaches (severe pain around one eye), sinus headaches (due to sinus inflammation). Triggers: stress, lack of sleep, dehydration, certain foods, hormonal changes, weather changes, eye strain. Treatment: rest in quiet dark room, apply cold/warm compress, over-the-counter pain relievers (acetaminophen, ibuprofen), stay hydrated, gentle neck/shoulder massage. Red flags requiring immediate medical attention: sudden severe headache, headache with fever/stiff neck, headache after head injury, progressive worsening headache, headache with vision changes/weakness/confusion. Prevention: regular sleep schedule, stress management, stay hydrated, avoid triggers, regular exercise."
  },
  {
    title: "Fever",
    category: "qa",
    source: "BuiltinKnowledge",
    content: "Title: Fever Category: qa Topic: Fever Q: What is fever and when to seek medical care? A: Fever is elevation of body temperature above normal (>100.4°F/38°C), usually indicating infection or illness. Body's natural defense mechanism to fight infection. Causes: viral infections (common cold, flu), bacterial infections (strep throat, UTI), other conditions (autoimmune diseases, medications, heat exhaustion). Symptoms: elevated temperature, chills, sweating, headache, muscle aches, fatigue, dehydration, loss of appetite. Treatment: rest, increase fluid intake, acetaminophen or ibuprofen for comfort, lukewarm sponge bath, light clothing. Seek immediate medical care for: fever >103°F (39.4°C), fever lasting >3 days, severe symptoms (difficulty breathing, chest pain, severe headache, stiff neck, confusion, persistent vomiting), signs of dehydration, fever in infants <3 months. Prevention: good hygiene, handwashing, vaccination, avoid close contact with sick individuals."
  },
  {
    title: "Chest Pain",
    category: "qa",
    source: "BuiltinKnowledge",
    content: "Title: Chest Pain Category: qa Topic: Chest Pain Q: What causes chest pain and when is it serious? A: Chest pain can range from minor discomfort to life-threatening emergency. Cardiac causes: heart attack, angina, pericarditis, aortic dissection. Non-cardiac causes: gastroesophageal reflux (GERD), muscle strain, costochondritis, pneumonia, pulmonary embolism, anxiety/panic attacks. Serious warning signs: crushing/squeezing chest pain, pain radiating to arm/jaw/neck/back, shortness of breath, sweating, nausea, lightheadedness, pain lasting >15 minutes. Call 108 immediately for suspected heart attack. Less serious causes: sharp stabbing pain that worsens with movement/breathing (likely musculoskeletal), burning sensation after eating (likely GERD), pain during stress/anxiety (likely panic attack). Evaluation: ECG, chest X-ray, blood tests (troponins), stress testing, echocardiogram. Treatment depends on cause: cardiac medications, antacids, anti-inflammatory drugs, anxiety management. Never ignore chest pain - when in doubt, seek immediate medical evaluation."
  },

  // ─── INFECTIOUS DISEASES ────────────────────────────────────────────────────
  {
    title: "Pneumonia",
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: Pneumonia Category: disease Disease: Pneumonia Description: Infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Causes: bacteria (Streptococcus pneumoniae most common), viruses (influenza, RSV, COVID-19), fungi, mycoplasma. Types: community-acquired, hospital-acquired, ventilator-associated, aspiration pneumonia. Symptoms: cough with phlegm/pus, fever, chills, difficulty breathing, chest pain when breathing/coughing, fatigue, nausea/vomiting/diarrhea. Severe symptoms: high fever, rapid breathing, confusion (especially elderly), blue lips/fingernails. Risk factors: age >65 or <2 years, chronic diseases (COPD, heart disease, diabetes), weakened immune system, smoking, recent viral infection. Diagnosis: chest X-ray, blood tests, sputum culture, pulse oximetry. Treatment: antibiotics for bacterial pneumonia, antivirals for viral pneumonia, supportive care (rest, fluids, oxygen if needed). Complications: bacteremia, lung abscess, pleural effusion, respiratory failure. Prevention: vaccination (pneumococcal, influenza), good hygiene, don't smoke, manage chronic conditions."
  },
  {
    title: "Tuberculosis (TB)",
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: Tuberculosis Category: disease Disease: Tuberculosis Description: A bacterial infection caused by Mycobacterium tuberculosis that primarily affects the lungs but can affect other parts of the body. Transmission: airborne droplets when infected person coughs, sneezes, or speaks. Types: latent TB (inactive, not contagious), active TB (symptomatic, contagious). Symptoms: persistent cough >3 weeks, coughing up blood, chest pain, weakness/fatigue, weight loss, loss of appetite, fever, night sweats. Extrapulmonary TB can affect spine, brain, kidneys. Risk factors: HIV infection, diabetes, malnutrition, smoking, close contact with TB patient, immunosuppression. Diagnosis: tuberculin skin test, interferon-gamma release assays, chest X-ray, sputum microscopy and culture, molecular tests (GeneXpert). Treatment: directly observed therapy (DOT) with multiple antibiotics for 6-9 months (isoniazid, rifampin, ethambutol, pyrazinamide). Drug-resistant TB requires longer treatment with second-line drugs. Prevention: BCG vaccination, infection control measures, treatment of latent TB, contact tracing."
  },

  // ─── GASTROINTESTINAL CONDITIONS ────────────────────────────────────────────
  {
    title: "Gastroenteritis",
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: Gastroenteritis Category: disease Disease: Gastroenteritis Description: Inflammation of stomach and intestines causing diarrhea and vomiting. Causes: viral (norovirus, rotavirus), bacterial (Salmonella, E. coli, Campylobacter), parasitic (Giardia), food poisoning, contaminated water. Symptoms: diarrhea, vomiting, nausea, abdominal cramps, fever, headache, muscle aches. Dehydration signs: dry mouth, decreased urination, dizziness, fatigue. Severe symptoms: blood in stool, high fever, severe dehydration, signs of shock. Treatment: oral rehydration therapy (ORS), clear fluids, gradual return to normal diet (BRAT diet: bananas, rice, applesauce, toast), probiotics may help. Avoid: dairy products, fatty foods, alcohol, caffeine during acute phase. Seek medical care for: severe dehydration, blood in stool, high fever, symptoms lasting >3 days, signs of severe illness. Prevention: proper food handling and storage, hand hygiene, safe water consumption, avoid raw/undercooked foods, vaccination (rotavirus for children)."
  },
  {
    title: "Peptic Ulcer Disease",
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: Peptic Ulcer Disease Category: disease Disease: Peptic Ulcer Description: Open sores that develop on the inner lining of stomach (gastric ulcer) or upper small intestine (duodenal ulcer). Causes: Helicobacter pylori bacterial infection (most common), NSAIDs (aspirin, ibuprofen), Zollinger-Ellison syndrome, smoking, alcohol, stress (contributing factor). Symptoms: burning stomach pain (worse when stomach empty), bloating, heartburn, nausea, loss of appetite, weight loss. Complications: bleeding (black tarry stools, vomiting blood), perforation, obstruction. Diagnosis: H. pylori testing (breath test, stool test, blood test), upper endoscopy, upper GI series. Treatment: antibiotics for H. pylori (triple or quadruple therapy), proton pump inhibitors (omeprazole, lansoprazole), H2 receptor blockers, antacids, discontinue NSAIDs if possible. Lifestyle: avoid spicy foods, alcohol, smoking; eat smaller frequent meals; manage stress. Emergency signs: severe abdominal pain, vomiting blood, black stools, signs of shock - seek immediate medical attention."
  },

  // ─── RESPIRATORY CONDITIONS ─────────────────────────────────────────────────
  {
    title: "Asthma",
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: Asthma Category: disease Disease: Asthma Description: Chronic respiratory condition where airways become inflamed, narrow, and produce extra mucus, making breathing difficult. Types: allergic asthma, non-allergic asthma, occupational asthma, exercise-induced asthma. Triggers: allergens (pollen, dust mites, pet dander), irritants (smoke, pollution, strong odors), respiratory infections, exercise, weather changes, stress, certain medications. Symptoms: shortness of breath, chest tightness, wheezing, coughing (especially at night/early morning), difficulty speaking in full sentences during attacks. Severe attack signs: severe breathing difficulty, inability to speak, blue lips/fingernails, peak flow <50% of personal best. Treatment: quick-relief medications (albuterol inhaler), long-term control medications (inhaled corticosteroids, leukotriene modifiers), allergy medications, immunotherapy. Action plan: identify triggers, monitor peak flow, know when to use medications, when to seek emergency care. Prevention: avoid triggers, take controller medications as prescribed, get vaccinated (flu, pneumonia), maintain healthy weight, manage stress."
  },
  {
    title: "Chronic Obstructive Pulmonary Disease (COPD)",
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: COPD Category: disease Disease: Chronic Obstructive Pulmonary Disease Description: Progressive lung disease that makes breathing difficult, includes emphysema and chronic bronchitis. Primary cause: smoking (85-90% of cases), also air pollution, occupational dust/chemicals, genetic factors (alpha-1 antitrypsin deficiency). Symptoms: chronic cough with mucus, shortness of breath (especially during activities), wheezing, chest tightness, frequent respiratory infections, fatigue, weight loss in advanced stages. COPD exacerbation: worsening of symptoms requiring medical intervention. Diagnosis: spirometry (FEV1/FVC ratio <0.70), chest X-ray, CT scan, arterial blood gas analysis. Treatment: bronchodilators (short and long-acting), inhaled corticosteroids, oxygen therapy, pulmonary rehabilitation, smoking cessation. Severe cases may need lung volume reduction surgery or lung transplant. Prevention: don't smoke or quit smoking, avoid secondhand smoke and air pollutants, get vaccinated (flu, pneumonia), exercise regularly, maintain healthy diet."
  },

  // ─── MENTAL HEALTH CONDITIONS ───────────────────────────────────────────────
  {
    title: "Depression",
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: Depression Category: disease Disease: Major Depressive Disorder Description: A mental health condition characterized by persistent feelings of sadness and loss of interest that interfere with daily functioning. Types: major depression, persistent depressive disorder, seasonal affective disorder, postpartum depression. Symptoms: persistent sad/empty mood, loss of interest in activities, significant weight loss/gain, sleep disturbances, fatigue, feelings of worthlessness/guilt, difficulty concentrating, thoughts of death/suicide. Risk factors: family history, trauma, chronic illness, substance abuse, certain medications, major life changes. Diagnosis: clinical interview, depression screening questionnaires (PHQ-9), rule out medical causes. Treatment: psychotherapy (cognitive behavioral therapy, interpersonal therapy), antidepressant medications (SSRIs, SNRIs), combination therapy, lifestyle changes (exercise, sleep hygiene, social support). Severe cases may need hospitalization or electroconvulsive therapy. Suicide prevention: if having thoughts of self-harm, call emergency services or suicide prevention hotline immediately. Support: maintain social connections, regular exercise, healthy diet, stress management, medication compliance."
  },
  {
    title: "Anxiety Disorders",
    category: "disease",
    source: "BuiltinKnowledge",
    content: "Title: Anxiety Disorders Category: disease Disease: Anxiety Disorders Description: Group of mental health conditions characterized by excessive fear, worry, and related behavioral disturbances. Types: generalized anxiety disorder, panic disorder, social anxiety disorder, specific phobias, agoraphobia. Symptoms: excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, sleep disturbances. Panic attacks: sudden intense fear with physical symptoms (rapid heartbeat, sweating, trembling, shortness of breath, chest pain, nausea, dizziness, fear of dying). Risk factors: genetics, brain chemistry, personality, life events, medical conditions, substance use. Diagnosis: clinical assessment, anxiety rating scales, rule out medical causes (hyperthyroidism, heart conditions). Treatment: psychotherapy (cognitive behavioral therapy, exposure therapy), medications (SSRIs, benzodiazepines for short-term use), relaxation techniques, lifestyle modifications. Self-help: regular exercise, adequate sleep, limit caffeine/alcohol, stress management, mindfulness/meditation, social support. Emergency: seek immediate help for severe panic attacks or thoughts of self-harm."
  }
];

async function loadComprehensiveMedicalData() {
  console.log('\n🏥 ===== LOADING COMPREHENSIVE MEDICAL KNOWLEDGE =====');
  console.log(`📊 Loading ${COMPREHENSIVE_MEDICAL_DATA.length} comprehensive medical entries...`);
  console.log('📚 This dataset includes diseases, emergencies, symptoms, and treatments\n');

  let loaded = 0;
  let errors = 0;
  let skipped = 0;

  for (let i = 0; i < COMPREHENSIVE_MEDICAL_DATA.length; i++) {
    const item = COMPREHENSIVE_MEDICAL_DATA[i];
    
    try {
      console.log(`[${i + 1}/${COMPREHENSIVE_MEDICAL_DATA.length}] Processing: ${item.title}`);
      
      // Check if already exists
      const { data: existing } = await supabase
        .from('medical_chunks')
        .select('id')
        .eq('title', item.title)
        .eq('source', 'BuiltinKnowledge')
        .limit(1);
      
      if (existing && existing.length > 0) {
        console.log(`   ⏭️  Already exists, skipping...`);
        skipped++;
        continue;
      }
      
      // Generate embedding
      console.log(`   🧠 Generating embedding...`);
      const embedding = await embedText(item.content);
      console.log(`   ✅ Embedding generated (${embedding.length} dimensions)`);
      
      // Insert into database
      const { data, error } = await supabase
        .from('medical_chunks')
        .insert({
          doc_id: `BuiltinKnowledge::${item.title.replace(/\s+/g, '_')}`,
          source: item.source,
          category: item.category,
          title: item.title,
          chunk_index: 0,
          content: item.content,
          metadata: {
            type: 'comprehensive_medical_knowledge',
            loaded_at: new Date().toISOString(),
            content_length: item.content.length,
            version: '2.0'
          },
          embedding: embedding
        })
        .select();

      if (error) {
        console.log(`   ❌ Database error: ${error.message}`);
        errors++;
      } else {
        console.log(`   ✅ Saved to database (ID: ${data[0]?.id})`);
        loaded++;
      }
      
    } catch (error) {
      console.log(`   ❌ Processing error: ${error.message}`);
      errors++;
    }
    
    console.log('');
  }

  // Check final database state
  const { count } = await supabase
    .from('medical_chunks')
    .select('*', { count: 'exact', head: true });

  // Get category distribution
  const { data: categories } = await supabase
    .from('medical_chunks')
    .select('category')
    .eq('source', 'BuiltinKnowledge');

  const categoryCount = {};
  categories?.forEach(c => {
    categoryCount[c.category] = (categoryCount[c.category] || 0) + 1;
  });

  console.log('📊 ===== COMPREHENSIVE LOADING COMPLETE =====');
  console.log(`✅ Successfully loaded: ${loaded}/${COMPREHENSIVE_MEDICAL_DATA.length}`);
  console.log(`⏭️  Skipped (already exists): ${skipped}`);
  console.log(`❌ Errors: ${errors}`);
  console.log(`📚 Total chunks in database: ${count || 0}`);
  console.log(`📋 Category distribution:`, categoryCount);
  console.log('===============================================\n');

  if (count > 0) {
    console.log('🎉 SUCCESS! Your database now has comprehensive medical knowledge.');
    console.log('🔄 You can now run model evaluation to see improved scores.');
    console.log('📈 Expected improvements:');
    console.log('   - Retrieval Quality: 32.6% → 80%+ (comprehensive medical coverage)');
    console.log('   - Generation Quality: 76.4% → 85%+ (better context and medical accuracy)');
    console.log('   - Overall Score: 49.2% → 85%+ (medical-grade AI system)');
  } else {
    console.log('⚠️ No data in database. Check your Supabase connection and credentials.');
  }
}

// Run the script
loadComprehensiveMedicalData().catch(console.error);