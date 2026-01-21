/**
 * Debug script to test ChatKit communication
 * This script will test the API endpoints directly to verify the backend is working
 */

const testQuery = "hi";
const sessionId = `test-session-${Date.now()}`;

async function testChatEndpoint() {
  console.log("Testing ChatKit API endpoints...\n");

  // Test the regular chat endpoint
  console.log("1. Testing regular chat endpoint:");
  try {
    const response = await fetch('https://aman778-rag-chatbot-backend.hf.space/api/v1/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: testQuery,
        session_id: sessionId,
        max_context: 5
      })
    });

    console.log(`   Status: ${response.status}`);
    const data = await response.json();
    console.log(`   Response:`, data);

    if (response.ok) {
      console.log(`   Answer: ${data.answer || 'No answer field'}`);
    }
  } catch (error) {
    console.error(`   Error calling regular chat endpoint:`, error.message);
  }

  console.log("\n2. Testing streaming chat endpoint:");
  try {
    const response = await fetch('https://aman778-rag-chatbot-backend.hf.space/api/v1/chat/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: testQuery,
        session_id: sessionId,
        max_context: 5
      })
    });

    console.log(`   Status: ${response.status}`);

    if (response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      console.log("   Streaming response:");

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              console.log(`     Received:`, data);
            } catch (e) {
              console.log(`     Raw line:`, line);
            }
          }
        }
      }

      // Process remaining buffer
      if (buffer.trim()) {
        console.log(`     Remaining buffer:`, buffer);
      }
    }
  } catch (error) {
    console.error(`   Error calling streaming chat endpoint:`, error.message);
  }

  console.log("\n3. Testing config endpoint:");
  try {
    const response = await fetch('https://aman778-rag-chatbot-backend.hf.space/api/v1/config');
    console.log(`   Status: ${response.status}`);
    const data = await response.json();
    console.log(`   Config:`, data);
  } catch (error) {
    console.error(`   Error calling config endpoint:`, error.message);
  }

  console.log("\nTest completed.");
}

// Run the test
testChatEndpoint().catch(console.error);