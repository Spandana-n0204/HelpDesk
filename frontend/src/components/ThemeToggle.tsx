import { useState, useEffect } from "react";
import { Moon, Sun } from "lucide-react";

export function ThemeToggle() {
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("dsce-theme");
    return saved ? saved === "dark" : true;
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("dsce-theme", dark ? "dark" : "light");
  }, [dark]);

  return (
    <button
      onClick={() => setDark(!dark)}
      className="p-2 rounded-lg bg-secondary hover:bg-sidebar-hover transition-colors"
      aria-label="Toggle theme"
    >
      {dark ? <Sun className="w-5 h-5 text-foreground" /> : <Moon className="w-5 h-5 text-foreground" />}
    </button>
  );
}
