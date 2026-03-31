import { useState, useCallback, useRef, useEffect } from "react";
import { AppSidebar } from "@/components/AppSidebar";
import { ThemeToggle } from "@/components/ThemeToggle";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessages, type Message } from "@/components/ChatMessages";
import { ChatInput } from "@/components/ChatInput";

const API_BASE = "http://127.0.0.1:8000";

// ── Device ID — identifies this browser as a "user" ──────────────────────────
function getDeviceId(): string {
  let id = localStorage.getItem("dsce_device_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("dsce_device_id", id);
  }
  return id;
}

// ── Types ─────────────────────────────────────────────────────────────────────
export interface Conversation {
  id:         string;
  title:      string;
  updated_at: string;
}

// ── Component ─────────────────────────────────────────────────────────────────
const Index = () => {
  const [messages,      setMessages]      = useState<Message[]>([]);
  const [loading,       setLoading]       = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const conversationId = useRef<string>("");
  const deviceId       = useRef<string>(getDeviceId());

  // Load sidebar conversations on mount
  useEffect(() => {
    fetchConversations();
  }, []);

  async function fetchConversations() {
    try {
      const res  = await fetch(`${API_BASE}/conversations?device_id=${deviceId.current}`);
      const data = await res.json();
      setConversations(data);
    } catch {
      // Sidebar just stays empty — not critical
    }
  }

  // ── Send message ─────────────────────────────────────────────────────────
  const handleSend = useCallback(async (text: string) => {
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question:        text,
          conversation_id: conversationId.current,
          device_id:       deviceId.current,
        }),
      });

      const data = await res.json();

      // Save conversation_id for follow-up messages
      if (data.conversation_id) {
        conversationId.current = data.conversation_id;
      }

      setMessages((prev) => [...prev, { role: "ai", content: data.answer }]);

      // Refresh sidebar to show new/updated conversation
      fetchConversations();

    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "ai", content: "Sorry, I couldn't reach the server. Please try again later." },
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  // ── New chat ──────────────────────────────────────────────────────────────
  const handleNewChat = () => {
    conversationId.current = "";
    setMessages([]);
  };

  // ── Load past conversation from sidebar ───────────────────────────────────
  const handleLoadConversation = async (conv: Conversation) => {
    try {
      const res  = await fetch(`${API_BASE}/conversations/${conv.id}/messages`);
      const data = await res.json();

      // Convert stored messages to Message format
      const loaded: Message[] = data.map((m: { role: string; content: string }) => ({
        role:    m.role === "user" ? "user" : "ai",
        content: m.content,
      }));

      conversationId.current = conv.id;
      setMessages(loaded);
    } catch {
      // If load fails, just start fresh
      handleNewChat();
    }
  };

  // ── Delete conversation ───────────────────────────────────────────────────
  const handleDeleteConversation = async (convId: string) => {
    try {
      await fetch(`${API_BASE}/conversations/${convId}`, { method: "DELETE" });
      // If we deleted the active conversation, clear the chat
      if (conversationId.current === convId) {
        handleNewChat();
      }
      fetchConversations();
    } catch {
      // ignore
    }
  };

  const hasMessages = messages.length > 0;

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <AppSidebar
        onNewChat={handleNewChat}
        conversations={conversations}
        onLoadConversation={handleLoadConversation}
        onDeleteConversation={handleDeleteConversation}
        activeConversationId={conversationId.current}
      />
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-3 border-b border-border">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">DSCE HelpDesk AI</span>
            <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-primary/10 text-primary">BETA</span>
          </div>
          <ThemeToggle />
        </header>

        {/* Main area */}
        {hasMessages ? (
          <ChatMessages messages={messages} loading={loading} />
        ) : (
          <WelcomeScreen />
        )}

        <ChatInput onSend={handleSend} disabled={loading} />
      </div>
    </div>
  );
};

export default Index;