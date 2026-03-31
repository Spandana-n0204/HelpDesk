// frontend/src/pages/Chat.tsx
// DSCE HelpDesk — Main chat page
// Drop this file into frontend/src/pages/Chat.tsx

import { useState, useRef, useEffect } from "react";

const API_BASE = "http://127.0.0.1:8000";

// ── Types ─────────────────────────────────────────────────────────────────────
interface Message {
  role: "user" | "bot";
  text: string;
}

// ── Session helpers ───────────────────────────────────────────────────────────
function getSessionId(): string {
  return localStorage.getItem("dsce_session_id") || "";
}
function saveSessionId(id: string) {
  localStorage.setItem("dsce_session_id", id);
}
function clearSessionId() {
  localStorage.removeItem("dsce_session_id");
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "bot",
      text: "Hi! I'm the DSCE HelpDesk assistant. Ask me about admissions, fees, courses, documents, or campus life.",
    },
  ]);
  const [input, setInput]     = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef             = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // ── Send ───────────────────────────────────────────────────────────────────
  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          session_id: getSessionId(),
        }),
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);

      const data = await res.json();
      saveSessionId(data.session_id);
      setMessages((prev) => [...prev, { role: "bot", text: data.answer }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: "Something went wrong. Please make sure the backend server is running.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  // ── Clear chat ─────────────────────────────────────────────────────────────
  async function clearChat() {
    const sid = getSessionId();
    if (sid) {
      try {
        await fetch(`${API_BASE}/session/${sid}`, { method: "DELETE" });
      } catch {
        // ignore
      }
    }
    clearSessionId();
    setMessages([
      {
        role: "bot",
        text: "Chat cleared. How can I help you?",
      },
    ]);
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-blue-700 text-white px-6 py-4 flex items-center justify-between shadow-md">
        <div>
          <h1 className="text-xl font-bold">DSCE HelpDesk</h1>
          <p className="text-blue-200 text-sm">Dayananda Sagar College of Engineering</p>
        </div>
        <button
          onClick={clearChat}
          className="text-sm bg-blue-600 hover:bg-blue-500 px-3 py-1.5 rounded-lg transition"
        >
          New Chat
        </button>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            {/* Avatar for bot */}
            {msg.role === "bot" && (
              <div className="w-8 h-8 rounded-full bg-blue-700 text-white flex items-center justify-center text-xs font-bold mr-2 mt-1 shrink-0">
                D
              </div>
            )}
            <div
              className={`max-w-[75%] px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-blue-600 text-white rounded-br-sm"
                  : "bg-white text-gray-800 rounded-bl-sm border border-gray-100"
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}

        {/* Typing indicator */}
        {loading && (
          <div className="flex justify-start">
            <div className="w-8 h-8 rounded-full bg-blue-700 text-white flex items-center justify-center text-xs font-bold mr-2 mt-1 shrink-0">
              D
            </div>
            <div className="bg-white border border-gray-100 px-4 py-3 rounded-2xl rounded-bl-sm shadow-sm">
              <div className="flex gap-1 items-center h-4">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="bg-white border-t border-gray-200 px-4 py-3">
        <div className="flex items-center gap-2 max-w-3xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
            placeholder="Ask about admissions, fees, courses…"
            disabled={loading}
            className="flex-1 border border-gray-300 rounded-xl px-4 py-2.5 text-sm outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-400 transition"
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="bg-blue-700 hover:bg-blue-600 disabled:bg-blue-300 text-white px-5 py-2.5 rounded-xl text-sm font-medium transition"
          >
            Send
          </button>
        </div>
        <p className="text-center text-xs text-gray-400 mt-2">
          Powered by DSCE HelpDesk AI · Responses may not always be accurate
        </p>
      </div>
    </div>
  );
}