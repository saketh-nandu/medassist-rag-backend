/**
 * load-sample-data.js — Load Sample Medical Knowledge
 * Simple script to populate database with sample medical data
 */

import { createClient } from '@supabase/supabase-js';
import { embedText } from './embed.js';
import dotenv from 'dotenv';
dotenv.config();

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// Sample medical knowledge
const SAMPLE_DATA = [
  {
    title: "Dengue Fever",
    category: "disease",
    content: "Title: Dengue Fever Category: disease Disease: Dengue Fever Description: A mosquito-borne viral infection causing high fever, severe headache, pain behind eyes, joint and muscle pain, and rash. Symptoms: high fever, severe headache, pain behind eyes, joint pain, muscle pain, rash, nausea, vomiting Precautions: Use mosquito repellent. Wear long sleeves. Eliminate standing water. Stay hydrated. Avoid aspirin and ibuprofen — use paracetamol only"
  },
  {
    title: "Malaria", 
    category: "disease",
    content: "Title: Malaria Category: disease Disease: Malaria Description: A parasitic disease transmitted by infected mosquitoes, causing cyclical fever, chills, and flu-like symptoms. Symptoms: cyclical fever, chills, sweating, headache, muscle pain, fatigue, nausea, vomiting Precautions: Use bed nets. Take antimalarial medication. Use insect repellent. Wear protective clothing. Eliminate standing water"
  },
  {
    title: "Typhoid",
    category: "disease", 
    content: "Title: Typhoid Category: disease Disease: Typhoid Description: A bacterial infection caused by Salmonella typhi, spread through contaminated food and water. Symptoms: sustained fever, weakness, stomach pain, headache, diarrhea or constipation, loss of appetite Precautions: Drink only boiled or bottled water. Eat freshly cooked food. Wash hands frequently. Get typhoid vaccine. Avoid raw vegetables in endemic areas"
  },
  {
    title: "Diabetes Type 2",
    category: "disease",
    content: "Title: Diabetes Type 2 Category: disease Disease: Diabetes Type 2 Description: A chronic condition affecting blood sugar regulation due to insulin resistance. Symptoms: increased thirst, frequent urination, increased hunger, fatigue, blurred vision, slow healing wounds, frequent infections Precautions: Maintain healthy diet. Exercise regularly. Monitor blood sugar. Take prescribed medications. Maintain healthy weight. Regular medical checkups"
  },
  {
    title: "Hypertension",
    category: "disease",
    content: "Title: Hypertension Category: disease Disease: Hypertension Description: High blood pressure, often called the 'silent killer' as it usually has no symptoms. Symptoms: usually no symptoms, sometimes headache, dizziness, nosebleeds in severe cases Precautions: Reduce salt intake. Exercise regularly. Maintain healthy weight. Limit alcohol. Don't smoke. Manage stress. Take prescribed medications"
  },
  {
    title: "Headache",
    category: "qa",
    content: "Title: Headache Category: qa Topic: Headache Q: What is (are) Headache ? A: Almost everyone has had a headache. Headache is the most common form of pain. It's a major reason people miss days at work or school or visit the doctor. The most common type of headache is a tension headache. Tension headaches are due to tight muscles in your shoulders, neck, scalp and jaw."
  },
  {
    title: "Chicken pox",
    category: "disease",
    content: "Title: Chicken pox Category: disease Disease: Chicken pox Description: Chickenpox is a highly contagious viral infection causing an itchy rash with small, fluid-filled blisters. Symptoms: itchy rash, fever, headache, tiredness, loss of appetite Precautions: Get vaccinated. Avoid contact with infected persons. Maintain good hygiene. Keep fingernails short to prevent scratching"
  },
  {
    title: "Common Cold",
    category: "disease",
    content: "Title: Common Cold Category: disease Disease: Common Cold Description: The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Symptoms: runny nose, congestion, cough, sore throat, sneezing, low-grade fever Precautions: Wash hands frequently. Avoid close contact with sick people. Don't touch your face. Get adequate sleep. Stay hydrated"
  }
];

async function loadSampleData() {
  console.log('\n🏥 ===== LOADING SAMPLE MEDICAL DATA =====');
  console.log(`📊 Loading ${SAMPLE_DATA.length} medical knowledge entries...\n`);

  let loaded = 0;
  let errors = 0;

  for (let i = 0; i < SAMPLE_DATA.length; i++) {
    const item = SAMPLE_DATA[i];
    
    try {
      console.log(`[${i + 1}/${SAMPLE_DATA.length}] Processing: ${item.title}`);
      
      // Generate embedding
      const embedding = await embedText(item.content);
      console.log(`   ✅ Embedding generated (${embedding.length} dimensions)`);
      
      // Insert into database
      const { data, error } = await supabase
        .from('medical_chunks')
        .insert({
          doc_id: `BuiltinKnowledge::${item.title.replace(/\s+/g, '_')}`,
          source: 'BuiltinKnowledge',
          category: item.category,
          title: item.title,
          chunk_index: 0,
          content: item.content,
          metadata: {
            type: 'medical_knowledge',
            loaded_at: new Date().toISOString()
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

  console.log('📊 ===== LOADING COMPLETE =====');
  console.log(`✅ Successfully loaded: ${loaded}/${SAMPLE_DATA.length}`);
  console.log(`❌ Errors: ${errors}`);
  console.log(`📚 Total chunks in database: ${count || 0}`);
  console.log('===============================\n');

  if (count > 0) {
    console.log('🎉 SUCCESS! Your database now has medical knowledge.');
    console.log('🔄 You can now run model evaluation to see scores.');
  } else {
    console.log('⚠️ No data in database. Check your Supabase connection and credentials.');
  }
}

// Run the script
loadSampleData().catch(console.error);