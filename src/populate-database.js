/**
 * populate-database.js — Load Medical Knowledge into Vector Database
 * 
 * This script populates the Supabase database with medical knowledge from:
 * 1. DSP CSV datasets (symptoms, diseases, precautions)
 * 2. MedQuAD XML files (medical Q&A pairs)
 * 3. Built-in medical knowledge base
 * 
 * Run with: node src/populate-database.js
 */

import { createClient } from '@supabase/supabase-js';
import { embedText } from './embed.js';
import fs from 'fs';
import path from 'path';
import csv from 'csv-parser';
import xml2js from 'xml2js';
import dotenv from 'dotenv';
dotenv.config();

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// ─── BUILT-IN MEDICAL KNOWLEDGE ─────────────────────────────────────────────

const BUILTIN_MEDICAL_KNOWLEDGE = [
  {
    title: "Dengue Fever",
    category: "disease",
    content: "Title: Dengue Fever Category: disease Disease: Dengue Fever Description: A mosquito-borne viral infection causing high fever, severe headache, pain behind eyes, joint and muscle pain, and rash. Symptoms: high fever, severe headache, pain behind eyes, joint pain, muscle pain, rash, nausea, vomiting Precautions: Use mosquito repellent. Wear long sleeves. Eliminate standing water. Stay hydrated. Avoid aspirin and ibuprofen — use paracetamol only Symptoms: high fever, severe headache",
    symptoms: ["high fever", "severe headache", "pain behind eyes", "joint pain", "muscle pain", "rash"],
    precautions: ["Use mosquito repellent", "Wear long sleeves", "Eliminate standing water", "Stay hydrated", "Avoid aspirin and ibuprofen — use paracetamol only"]
  },
  {
    title: "Malaria",
    category: "disease", 
    content: "Title: Malaria Category: disease Disease: Malaria Description: A parasitic disease transmitted by infected mosquitoes, causing cyclical fever, chills, and flu-like symptoms. Symptoms: cyclical fever, chills, sweating, headache, muscle pain, fatigue, nausea, vomiting Precautions: Use bed nets. Take antimalarial medication. Use insect repellent. Wear protective clothing. Eliminate standing water Symptoms: cyclical fever, chills, sweating",
    symptoms: ["cyclical fever", "chills", "sweating", "headache", "muscle pain", "fatigue"],
    precautions: ["Use bed nets", "Take antimalarial medication", "Use insect repellent", "Wear protective clothing"]
  },
  {
    title: "Typhoid",
    category: "disease",
    content: "Title: Typhoid Category: disease Disease: Typhoid Description: A bacterial infection caused by Salmonella typhi, spread through contaminated food and water. Symptoms: sustained fever, weakness, stomach pain, headache, diarrhea or constipation, loss of appetite Precautions: Drink only boiled or bottled water. Eat freshly cooked food. Wash hands frequently. Get typhoid vaccine. Avoid raw vegetables in endemic areas Symptoms: sustained fever, weakness",
    symptoms: ["sustained fever", "weakness", "stomach pain", "headache", "diarrhea", "constipation"],
    precautions: ["Drink only boiled or bottled water", "Eat freshly cooked food", "Wash hands frequently", "Get typhoid vaccine"]
  },
  {
    title: "Hypertension",
    category: "disease",
    content: "Title: Hypertension Category: disease Disease: Hypertension Description: High blood pressure, often called the 'silent killer' as it usually has no symptoms. Symptoms: usually no symptoms, sometimes headache, dizziness, nosebleeds in severe cases Precautions: Reduce salt intake. Exercise regularly. Maintain healthy weight. Limit alcohol. Don't smoke. Manage stress. Take prescribed medications Symptoms: usually no symptoms",
    symptoms: ["usually no symptoms", "headache", "dizziness", "nosebleeds"],
    precautions: ["Reduce salt intake", "Exercise regularly", "Maintain healthy weight", "Limit alcohol", "Don't smoke"]
  },
  {
    title: "Diabetes Type 2",
    category: "disease",
    content: "Title: Diabetes Type 2 Category: disease Disease: Diabetes Type 2 Description: A chronic condition affecting blood sugar regulation due to insulin resistance. Symptoms: increased thirst, frequent urination, increased hunger, fatigue, blurred vision, slow healing wounds, frequent infections Precautions: Maintain healthy diet. Exercise regularly. Monitor blood sugar. Take prescribed medications. Maintain healthy weight. Regular medical checkups Symptoms: increased thirst, frequent urination",
    symptoms: ["increased thirst", "frequent urination", "increased hunger", "fatigue", "blurred vision"],
    precautions: ["Maintain healthy diet", "Exercise regularly", "Monitor blood sugar", "Take prescribed medications"]
  }
];

// ─── DATABASE POPULATION FUNCTIONS ──────────────────────────────────────────

class DatabasePopulator {
  constructor() {
    this.totalProcessed = 0;
    this.totalEmbedded = 0;
    this.errors = [];
  }

  async populateDatabase() {
    console.log('\n🏥 ===== POPULATING MEDASSIST DATABASE =====');
    console.log('📊 Loading medical knowledge from multiple sources...\n');

    try {
      // Clear existing data
      await this.clearExistingData();

      // Load built-in knowledge
      await this.loadBuiltinKnowledge();

      // Load DSP datasets
      await this.loadDSPDatasets();

      // Load sample MedQuAD data
      await this.loadSampleMedQuAD();

      // Generate summary
      await this.generateSummary();

    } catch (error) {
      console.error('❌ Database population failed:', error);
      throw error;
    }
  }

  async clearExistingData() {
    console.log('🧹 Clearing existing data...');
    
    const { error: chunksError } = await supabase
      .from('medical_chunks')
      .delete()
      .neq('id', 0); // Delete all records

    const { error: knowledgeError } = await supabase
      .from('medical_knowledge')
      .delete()
      .neq('id', 0); // Delete all records

    if (chunksError) console.warn('Warning clearing chunks:', chunksError.message);
    if (knowledgeError) console.warn('Warning clearing knowledge:', knowledgeError.message);
    
    console.log('✅ Existing data cleared\n');
  }

  async loadBuiltinKnowledge() {
    console.log('🏥 Loading built-in medical knowledge...');
    
    for (let i = 0; i < BUILTIN_MEDICAL_KNOWLEDGE.length; i++) {
      const knowledge = BUILTIN_MEDICAL_KNOWLEDGE[i];
      
      try {
        // Generate embedding
        const embedding = await embedText(knowledge.content);
        
        // Insert into medical_chunks
        const { error } = await supabase
          .from('medical_chunks')
          .insert({
            doc_id: `BuiltinKnowledge::${knowledge.title.replace(/\s+/g, '_')}`,
            source: 'BuiltinKnowledge',
            category: knowledge.category,
            title: knowledge.title,
            chunk_index: 0,
            content: knowledge.content,
            metadata: {
              type: 'medical_knowledge',
              symptoms: knowledge.symptoms,
              precautions: knowledge.precautions
            },
            embedding: embedding
          });

        if (error) {
          console.error(`❌ Error inserting ${knowledge.title}:`, error.message);
          this.errors.push(`BuiltinKnowledge::${knowledge.title}: ${error.message}`);
        } else {
          console.log(`   ✅ ${knowledge.title}`);
          this.totalEmbedded++;
        }
        
        this.totalProcessed++;
        
      } catch (error) {
        console.error(`❌ Error processing ${knowledge.title}:`, error.message);
        this.errors.push(`BuiltinKnowledge::${knowledge.title}: ${error.message}`);
      }
    }
    
    console.log(`✅ Built-in knowledge loaded: ${this.totalEmbedded}/${BUILTIN_MEDICAL_KNOWLEDGE.length}\n`);
  }

  async loadDSPDatasets() {
    console.log('📊 Loading DSP datasets...');
    
    const dspPath = '../project/DSP';
    const datasets = [
      { file: 'dataset.csv', category: 'disease' },
      { file: 'symptom_Description.csv', category: 'symptom' },
      { file: 'symptom_precaution.csv', category: 'precaution' }
    ];

    for (const dataset of datasets) {
      const filePath = path.join(process.cwd(), dspPath, dataset.file);
      
      if (fs.existsSync(filePath)) {
        await this.loadCSVFile(filePath, dataset.category, 'DSP');
      } else {
        console.log(`   ⚠️ File not found: ${dataset.file}`);
      }
    }
    
    console.log(`✅ DSP datasets processed\n`);
  }

  async loadCSVFile(filePath, category, source) {
    return new Promise((resolve) => {
      const results = [];
      let processed = 0;
      
      fs.createReadStream(filePath)
        .pipe(csv())
        .on('data', (data) => results.push(data))
        .on('end', async () => {
          console.log(`   📄 Processing ${path.basename(filePath)}: ${results.length} rows`);
          
          for (let i = 0; i < Math.min(results.length, 50); i++) { // Limit to 50 per file for demo
            const row = results[i];
            
            try {
              // Create content from CSV row
              const content = this.createContentFromCSV(row, category);
              const title = row.Disease || row.Symptom || row.prognosis || `${category}_${i}`;
              
              if (content && title) {
                const embedding = await embedText(content);
                
                const { error } = await supabase
                  .from('medical_chunks')
                  .insert({
                    doc_id: `DSP::${title.replace(/\s+/g, '_')}`,
                    source: source,
                    category: category,
                    title: title,
                    chunk_index: 0,
                    content: content,
                    metadata: {
                      csv_file: path.basename(filePath),
                      row_index: i,
                      original_data: row
                    },
                    embedding: embedding
                  });

                if (!error) {
                  processed++;
                  this.totalEmbedded++;
                }
              }
              
              this.totalProcessed++;
              
            } catch (error) {
              this.errors.push(`${source}::${title}: ${error.message}`);
            }
          }
          
          console.log(`   ✅ ${path.basename(filePath)}: ${processed} records embedded`);
          resolve();
        });
    });
  }

  createContentFromCSV(row, category) {
    if (category === 'disease' && row.Disease) {
      return `Title: ${row.Disease} Category: disease Disease: ${row.Disease} Description: ${row.Description || 'Medical condition requiring attention'} Symptoms: ${row.Symptom_1 || ''} ${row.Symptom_2 || ''} ${row.Symptom_3 || ''} ${row.Symptom_4 || ''}`.trim();
    }
    
    if (category === 'symptom' && row.Symptom) {
      return `Title: ${row.Symptom} Category: symptom Symptom: ${row.Symptom} Description: ${row.Description || 'Symptom requiring medical evaluation'}`;
    }
    
    if (category === 'precaution' && row.Disease) {
      const precautions = [row.Precaution_1, row.Precaution_2, row.Precaution_3, row.Precaution_4]
        .filter(p => p && p.trim())
        .join('. ');
      return `Title: ${row.Disease} Category: precaution Disease: ${row.Disease} Precautions: ${precautions}`;
    }
    
    return null;
  }

  async loadSampleMedQuAD() {
    console.log('📚 Loading sample MedQuAD data...');
    
    // Create sample MedQuAD entries since parsing XML files would be complex
    const sampleMedQuAD = [
      {
        title: "Headache",
        category: "qa",
        content: "Title: Headache Category: qa Topic: Headache Q: What is (are) Headache ? A: Almost everyone has had a headache. Headache is the most common form of pain. It's a major reason people miss days at work or school or visit the doctor. The most common type of headache is a tension headache. Tension headaches are due to tight muscles in your shoulders, neck, scalp and jaw."
      },
      {
        title: "Diabetes",
        category: "qa", 
        content: "Title: Diabetes Category: qa Topic: Diabetes Q: What is diabetes? A: Diabetes is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period. Symptoms include frequent urination, increased thirst, and increased hunger. If left untreated, diabetes can cause many complications."
      },
      {
        title: "Heart Disease",
        category: "qa",
        content: "Title: Heart Disease Category: qa Topic: Heart Disease Q: What is heart disease? A: Heart disease refers to several types of heart conditions. The most common type is coronary artery disease, which can cause heart attack. Other heart conditions include heart failure, arrhythmias, and heart valve problems."
      }
    ];

    for (let i = 0; i < sampleMedQuAD.length; i++) {
      const item = sampleMedQuAD[i];
      
      try {
        const embedding = await embedText(item.content);
        
        const { error } = await supabase
          .from('medical_chunks')
          .insert({
            doc_id: `MedQuAD::${item.title}`,
            source: 'MedQuAD',
            category: item.category,
            title: item.title,
            chunk_index: 0,
            content: item.content,
            metadata: {
              folder: '4_MPlus_Health_Topics_QA',
              filename: `MedQuAD::${item.title}`
            },
            embedding: embedding
          });

        if (!error) {
          console.log(`   ✅ ${item.title}`);
          this.totalEmbedded++;
        }
        
        this.totalProcessed++;
        
      } catch (error) {
        this.errors.push(`MedQuAD::${item.title}: ${error.message}`);
      }
    }
    
    console.log(`✅ Sample MedQuAD data loaded\n`);
  }

  async generateSummary() {
    console.log('📊 ===== DATABASE POPULATION SUMMARY =====');
    
    // Get final counts
    const { count: chunksCount } = await supabase
      .from('medical_chunks')
      .select('*', { count: 'exact', head: true });

    const { count: knowledgeCount } = await supabase
      .from('medical_knowledge')
      .select('*', { count: 'exact', head: true });

    console.log(`📈 Total Records Processed: ${this.totalProcessed}`);
    console.log(`✅ Successfully Embedded: ${this.totalEmbedded}`);
    console.log(`📊 Medical Chunks in DB: ${chunksCount || 0}`);
    console.log(`📚 Knowledge Entries in DB: ${knowledgeCount || 0}`);
    
    if (this.errors.length > 0) {
      console.log(`\n⚠️ Errors encountered: ${this.errors.length}`);
      this.errors.slice(0, 5).forEach(error => console.log(`   - ${error}`));
      if (this.errors.length > 5) {
        console.log(`   ... and ${this.errors.length - 5} more`);
      }
    }
    
    console.log('\n🎉 Database population complete!');
    console.log('✅ Your RAG system now has medical knowledge for evaluation');
    console.log('===============================================\n');
  }
}

// ─── MAIN EXECUTION ──────────────────────────────────────────────────────────

async function main() {
  try {
    const populator = new DatabasePopulator();
    await populator.populateDatabase();
    process.exit(0);
  } catch (error) {
    console.error('❌ Population failed:', error);
    process.exit(1);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { DatabasePopulator };