import { useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

export interface Message {
  role: "user" | "ai";
  content: string;
}

interface ChatMessagesProps {
  messages: Message[];
  loading: boolean;
}

export function ChatMessages({ messages, loading }: ChatMessagesProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
      {messages.map((msg, i) => (
        <div
          key={i}
          className={`chat-fade-in flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
        >
          <div
            className={`max-w-[70%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
              msg.role === "user"
                ? "bg-chat-user text-chat-user-fg rounded-br-md"
                : "bg-chat-ai text-chat-ai-fg rounded-bl-md"
            }`}
          >
            {msg.role === "ai" ? (
              <div className="prose prose-sm prose-invert max-w-none">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
            ) : (
              msg.content
            )}
          </div>
        </div>
      ))}

      {loading && (
        <div className="chat-fade-in flex justify-start">
          <div className="bg-chat-ai px-4 py-3 rounded-2xl rounded-bl-md flex gap-1.5">
            <span className="w-2 h-2 rounded-full bg-primary pulse-dot" />
            <span className="w-2 h-2 rounded-full bg-primary pulse-dot" style={{ animationDelay: "0.2s" }} />
            <span className="w-2 h-2 rounded-full bg-primary pulse-dot" style={{ animationDelay: "0.4s" }} />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
