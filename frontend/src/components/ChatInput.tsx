import { useState } from "react";
import { Send } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");

  const handleSend = () => {
    if (!value.trim() || disabled) return;
    onSend(value.trim());
    setValue("");
  };

  return (
    <div className="p-4 border-t border-border">
      <div className="glass-panel flex items-center gap-2 px-4 py-2">
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask anything about DSCE..."
          disabled={disabled}
          className="flex-1 bg-transparent outline-none text-sm text-foreground placeholder:text-muted-foreground"
        />
        <button
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className="p-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90 transition-opacity disabled:opacity-40"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
