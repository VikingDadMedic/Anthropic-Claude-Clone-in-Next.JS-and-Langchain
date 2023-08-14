// 1. Import dependencies
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { DynamicTool, DynamicStructuredTool, WikipediaQueryRun, SerpAPI } from "langchain/tools";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { StreamingTextResponse, AnthropicStream } from 'ai';
import * as z from 'zod';


export const runtime = 'edge';

// 2. Define interfaces
interface File {
  base64: string;
  content: string;
}
interface FunctionInfo {
  name: string;
  active: boolean;
}

function buildPrompt(
  messages: { content: string; role: 'system' | 'user' | 'assistant' }[]
) {
  return (
    messages
      .map(({ content, role }) => {
        if (role === 'user') {
          return `Human: ${content}`
        } else {
          return `Assistant: ${content}`
        }
      })
      .join('\n\n') + 'Assistant:'
  )
}

// 3. Set up environment variables
const privateKey: string = process.env.SUPABASE_PRIVATE_KEY!;
const url: string = process.env.SUPABASE_URL!;
if (!privateKey) throw new Error(`Expected env var SUPABASE_PRIVATE_KEY`);
if (!url) throw new Error(`Expected env var SUPABASE_URL`);

// 4. Define the POST function
export async function POST(req: Request, res: Response) {
  // 5. Extract data from the request
  const { messages, functions, files, selectedModel, selectedVectorStorage } = await req.json();

  // 6. Handle the 'claude-2-100k' model case
  if (selectedModel === 'claude-2') {
    // 7. Generate an example response for the Claude model
    const result = await fetch('https://api.anthropic.com/v1/complete', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY
      },
      body: JSON.stringify({
        prompt: buildPrompt(messages),
        model: 'claude-v1',
        max_tokens_to_sample: 300,
        temperature: 0.9,
        stream: true
      })
    })
   
    // Check for errors
    if (!result.ok) {
      return new Response(await result.text(), {
        status: result.status
      })
    }
    const stream = AnthropicStream(result)
    // const chunks: string[] = result.split(" ");
    const responseStream = new StreamingTextResponse(stream);
  } else {
    // 8. Process the input data
    const latestMessage: string = messages[messages.length - 1].content;
    const decodedFiles: File[] = files.map((file: { base64: string }) => {
      return {
        ...file,
        content: Buffer.from(file.base64, 'base64').toString('utf-8')
      };
    });
    let argForExecutor: string = latestMessage;
    if (files.length > 0) {
      // 9. Set up Supabase vector store for file content
      const client = createClient(url, privateKey);
      const string: string = decodedFiles.map((file) => file.content).join('\n');
      const vectorStore = await SupabaseVectorStore.fromTexts(
        [string],
        [],
        new OpenAIEmbeddings(),
        {
          client,
          tableName: "documents",
          queryName: "match_documents",
        }
      );
      // 10. Perform similarity search using vector store
      const vectorResultsArr = await vectorStore.similaritySearch(latestMessage, 1);
      const vectorResultsStr: string = vectorResultsArr.map((result) => result.pageContent).join('\n');
      argForExecutor = `USER QUERY: ${latestMessage} --- Before using prior knowledge base, use the following from new info: ${vectorResultsStr}`;
    }

    // 11. Set up agent executor with tools and model
    const model = new ChatOpenAI({ temperature: 0, streaming: true });
    const wikipediaQuery = new WikipediaQueryRun({
      topKResults: 1,
      maxDocContentLength: 1000,
    });
    // 11.5 Set up agent executor with tools and model
    const serpApiQuery = new SerpAPI(process.env.SERPAPI_API_KEY, {
      location: "United States",
      gl: "us",
      hl: "en",
      safe: "active",
      nfpr: "1"
    });

    // 12. Define a dynamic tool for returning the value of foo
    // const foo = new DynamicTool({
    //   name: 'foo',
    //   description: 'Returns the value of foo',
    //   func: async (): Promise<string> => {
    //     return 'The value of foo is "this is a langchain, next.js, supabase, claude, openai and AI demo"';
    //   }
    // });

    // 13. Define a dynamic structured tool for fetching crypto price
    const fetchDestinationGuide = new DynamicStructuredTool({
      name: 'fetchDestinationGuide',
      description: 'Fetches and returns a destination guide for a specified city',
      schema: z.object({
        cityISO: z.string(),
      }),
      func: async (options: { cityISO: string }): Promise<string> => {
        const { cityISO } = options;
        const url = `https://api.arrivalguides.com/api/xml/Travelguide?auth=7441604e8621acef46dc91746f25041f3b79d7b2&lang=en&iso=${cityISO}&v=13`;
        const response = await fetch(url);
        const data = await response.json();
        const guide = JSON.stringify(data.data.destination.description)
        return guide;
        // return data[cryptoName.toLowerCase()][vsCurrency!.toLowerCase()].toString();
      },
    });

    // 14. Define available functions and tools
    const availableFunctions: Record<string, any | DynamicStructuredTool> = {
      wikipediaQuery,
      serpApiQuery,
      fetchDestinationGuide,
      // foo
    };
    const tools: Array<any | DynamicStructuredTool> = [wikipediaQuery, serpApiQuery];
    if (functions) {
      functions.forEach((func: FunctionInfo) => {
        if (func.active) {
          tools.push(availableFunctions[func.name]);
        }
      });
    }

    // 15. Initialize agent executor with tools and model
    const executor = await initializeAgentExecutorWithOptions(tools, model, {
      agentType: "openai-functions",
    });

    // 16. Run the executor and return the result as a streaming response
    const result: string = await executor.run(argForExecutor);
    const chunks: string[] = result.split(" ");
    const responseStream = new ReadableStream({
      async start(controller) {
        for (const chunk of chunks) {
          const bytes = new TextEncoder().encode(chunk + " ");
          controller.enqueue(bytes);
          await new Promise((r) => setTimeout(r, Math.floor(Math.random() * 20 + 10)));
        }
        controller.close();
      },
    });
    return new StreamingTextResponse(responseStream);
  }
}
