"use client";

import { useState } from "react";

interface Message {
  role: "user" | "ai";
  content: string;
  sources?: string[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg.content }),
      });

      if (!res.ok) {
        throw new Error("Failed to fetch response");
      }

      const data = await res.json();
      const aiMsg: Message = {
        role: "ai",
        content: data.answer,
        sources: data.sources,
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { role: "ai", content: "Error: Could not get response from backend." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-24 bg-gray-900 text-white">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold mb-8 text-center w-full">RAG Chatbot</h1>
      </div>

      <div className="flex-1 w-full max-w-3xl overflow-y-auto mb-8 bg-gray-800 rounded-lg p-4 space-y-4 min-h-[50vh]">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-10">
            Ask a question about your documents!
          </div>
        )}
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"
              }`}
          >
            <div
              className={`p-3 rounded-lg max-w-[80%] ${msg.role === "user"
                ? "bg-blue-600 text-white"
                : "bg-gray-700 text-gray-200"
                }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-2 text-xs text-gray-400 border-t border-gray-600 pt-1">
                  Sources: {msg.sources.join(", ")}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex flex-col items-start">
            <div className="p-3 rounded-lg bg-gray-700 text-gray-200 animate-pulse">
              Thinking...
            </div>
          </div>
        )}
      </div>

      <div className="w-full max-w-3xl flex gap-2">
        <input
          type="text"
          className="flex-1 p-3 rounded-lg bg-gray-800 border border-gray-700 focus:outline-none focus:border-blue-500 text-white"
          placeholder="Type your question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          className="p-3 bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 font-bold"
        >
          Send
        </button>
      </div>
    </main>
  );
}
