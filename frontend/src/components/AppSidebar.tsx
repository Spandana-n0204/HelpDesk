import { MessageSquarePlus, MessagesSquare, Library, Settings, Trash2 } from "lucide-react";
import { Conversation } from "@/pages/Index";

interface SidebarProps {
  onNewChat:             () => void;
  conversations:         Conversation[];
  onLoadConversation:    (conv: Conversation) => void;
  onDeleteConversation:  (id: string) => void;
  activeConversationId:  string;
}

const navItems = [
  { icon: MessageSquarePlus, label: "New Chat",     action: "new"     },
  { icon: Library,           label: "Library",      action: "library"  },
  { icon: Settings,          label: "Settings",     action: "settings" },
];

export function AppSidebar({
  onNewChat,
  conversations,
  onLoadConversation,
  onDeleteConversation,
  activeConversationId,
}: SidebarProps) {
  return (
    <aside className="w-64 h-screen flex flex-col bg-sidebar border-r border-sidebar-border shrink-0">
      {/* Logo */}
      <div className="p-5">
        <h1 className="text-xl font-bold text-foreground tracking-tight">
          <span className="text-primary">⬡</span> DSCE HelpDesk
        </h1>
      </div>

      {/* Top nav */}
      <nav className="px-3 space-y-1">
        {navItems.map((item) => (
          <button
            key={item.label}
            onClick={item.action === "new" ? onNewChat : undefined}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-muted-foreground hover:bg-sidebar-hover hover:text-foreground transition-all duration-200"
          >
            <item.icon className="w-4 h-4" />
            {item.label}
          </button>
        ))}
      </nav>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto px-3 mt-4">
        <div className="flex items-center gap-2 px-3 mb-2">
          <MessagesSquare className="w-4 h-4 text-muted-foreground" />
          <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Recent Chats
          </span>
        </div>

        {conversations.length === 0 ? (
          <p className="text-xs text-muted-foreground px-3 py-2">
            No conversations yet.
          </p>
        ) : (
          <div className="space-y-0.5">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-all duration-200 ${
                  activeConversationId === conv.id
                    ? "bg-primary/10 text-foreground"
                    : "text-muted-foreground hover:bg-sidebar-hover hover:text-foreground"
                }`}
                onClick={() => onLoadConversation(conv)}
              >
                <span className="flex-1 text-sm truncate">{conv.title}</span>
                {/* Delete button — only visible on hover */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConversation(conv.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:text-destructive transition-all"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-sidebar-border">
        <p className="text-xs text-muted-foreground">DSCE HelpDesk AI v1.0</p>
      </div>
    </aside>
  );
}
