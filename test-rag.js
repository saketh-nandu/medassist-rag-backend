/**
 * Quick test script for the enhanced RAG pipeline
 */

const testQuery = async (message) => {
  console.log(`\n${'='.repeat(80)}`);
  console.log(`QUERY: "${message}"`);
  console.log('='.repeat(80));

  const response = await fetch('http://localhost:3001/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });

  const data = await response.json();

  console.log(`\n✅ Response received (${data.severity} severity)`);
  console.log(`\n📋 Conditions: ${data.conditions.join(', ') || 'None'}`);
  console.log(`\n🎯 Affected Areas: ${data.affectedAreas.join(', ') || 'None'}`);
  console.log(`\n💬 Content:\n${data.content}`);
  console.log(`\n📌 Suggested Actions:`);
  data.suggestedActions.forEach((action, i) => console.log(`   ${i + 1}. ${action}`));
};

// Test queries
(async () => {
  try {
    await testQuery('I have a severe headache and fever for 3 days');
    await testQuery('What causes diabetes?');
    await testQuery('I feel dizzy when I stand up');
  } catch (err) {
    console.error('❌ Error:', err.message);
  }
})();
