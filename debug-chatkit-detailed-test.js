/**
 * Detailed debug script to test ChatKit communication and API endpoints
 * This script will test both the frontend and backend communication in detail
 */

const testQuery = "hi";
const sessionId = `test-session-${Date.now()}`;

async function detailedTestChatEndpoint() {
  console.log("=== Detailed ChatKit API Endpoint Test ===\n");

  console.log("Testing with query:", testQuery);
  console.log("Session ID:", sessionId);
  console.log("Timestamp:", new Date().toISOString());

  // Test the regular chat endpoint
  console.log("\n1. Testing regular chat endpoint:");
  console.log("   POST https://aman778-rag-chatbot-backend.hf.space/api/v1/chat");
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
    console.log(`   Status Text: ${response.statusText}`);

    const headers = {};
    for (const [key, value] of response.headers.entries()) {
      headers[key] = value;
    }
    console.log(`   Headers:`, headers);

    const data = await response.json();
    console.log(`   Response Body:`, JSON.stringify(data, null, 2));

    if (response.ok) {
      console.log(`   ✓ Success: Answer received`);
      console.log(`   Answer: ${data.answer || 'No answer field'}`);
      console.log(`   Sources: ${data.sources ? data.sources.length : 0} sources`);
    } else {
      console.log(`   ✗ Error: ${response.status} - ${response.statusText}`);
    }
  } catch (error) {
    console.error(`   ✗ Network Error:`, error.message);
  }

  console.log("\n2. Testing streaming chat endpoint:");
  console.log("   POST https://aman778-rag-chatbot-backend.hf.space/api/v1/chat/stream");
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
    console.log(`   Status Text: ${response.statusText}`);

    const headers = {};
    for (const [key, value] of response.headers.entries()) {
      headers[key] = value;
    }
    console.log(`   Headers:`, headers);

    if (response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      console.log("   Reading streaming response:");

      let chunkCount = 0;
      let receivedTokens = 0;
      let completeResponse = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          console.log("   ✅ Stream completed");
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() === '') continue;

          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              chunkCount++;
              console.log(`     Chunk ${chunkCount}:`, JSON.stringify(data, null, 2));

              if (data.type === 'token' && data.content) {
                receivedTokens++;
                completeResponse += data.content;
                console.log(`       Token #${receivedTokens}: "${data.content}"`);
              } else if (data.type === 'complete') {
                console.log(`       Stream completion received`);
              } else if (data.type === 'error') {
                console.log(`       ❌ Error received:`, data.message);
              } else if (data.type === 'source') {
                console.log(`       Source received:`, data.chunk_id);
              }
            } catch (e) {
              console.log(`     Raw line:`, line);
              console.log(`     ⚠️  Malformed JSON:`, e.message);
            }
          } else {
            console.log(`     Raw line:`, line);
          }
        }
      }

      // Process remaining buffer
      if (buffer.trim()) {
        console.log(`     Remaining buffer:`, buffer);
      }

      console.log(`   Total chunks received: ${chunkCount}`);
      console.log(`   Total tokens received: ${receivedTokens}`);
      console.log(`   Complete response: "${completeResponse}"`);
    } else {
      console.log("   ❌ No response body for streaming");
    }
  } catch (error) {
    console.error(`   ✗ Network Error:`, error.message);
  }

  console.log("\n3. Testing config endpoint:");
  console.log("   GET https://aman778-rag-chatbot-backend.hf.space/api/v1/config");
  try {
    const response = await fetch('https://aman778-rag-chatbot-backend.hf.space/api/v1/config');
    console.log(`   Status: ${response.status}`);
    console.log(`   Status Text: ${response.statusText}`);

    const headers = {};
    for (const [key, value] of response.headers.entries()) {
      headers[key] = value;
    }
    console.log(`   Headers:`, headers);

    const data = await response.json();
    console.log(`   Config:`, JSON.stringify(data, null, 2));
  } catch (error) {
    console.error(`   ✗ Network Error:`, error.message);
  }

  console.log("\n4. Testing health endpoint:");
  console.log("   GET https://aman778-rag-chatbot-backend.hf.space/api/v1/health");
  try {
    const response = await fetch('https://aman778-rag-chatbot-backend.hf.space/api/v1/health');
    console.log(`   Status: ${response.status}`);
    console.log(`   Status Text: ${response.statusText}`);

    const headers = {};
    for (const [key, value] of response.headers.entries()) {
      headers[key] = value;
    }
    console.log(`   Headers:`, headers);

    const data = await response.json();
    console.log(`   Health:`, JSON.stringify(data, null, 2));
  } catch (error) {
    console.error(`   ✗ Network Error:`, error.message);
  }

  console.log("\n=== Test completed ===");
}

// Run the detailed test
detailedTestChatEndpoint().catch(console.error);