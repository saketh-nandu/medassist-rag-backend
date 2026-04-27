/**
 * setupDb.js
 * Tests Supabase connection and verifies the table exists.
 *
 * The table must be created manually in Supabase SQL Editor.
 * Run the SQL from setup.sql first, then run this to verify.
 *
 * Usage: node src/setupDb.js
 */

import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
dotenv.config();

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

async function verify() {
  console.log('🔌 Testing Supabase connection...');
  console.log(`   URL: ${process.env.SUPABASE_URL}\n`);

  // Test connection by querying the table
  const { data, error } = await supabase
    .from('medical_knowledge')
    .select('id')
    .limit(1);

  if (error) {
    if (error.message.includes('does not exist') || error.code === '42P01') {
      console.log('❌ Table "medical_knowledge" does not exist yet.');
      console.log('\n👉 Please run the SQL in setup.sql in your Supabase SQL Editor:');
      console.log('   https://supabase.com/dashboard/project/smybetwoumsiqdwpirvi/sql/new\n');
    } else {
      console.log('❌ Connection error:', error.message);
    }
    return false;
  }

  console.log('✅ Connected to Supabase!');
  console.log('✅ Table "medical_knowledge" exists and is ready.');
  console.log('\nNext step: node src/ingest.js');
  return true;
}

verify().catch(console.error);
