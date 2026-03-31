import aiOrb from "@/assets/ai-orb.png";

export function WelcomeScreen() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-6 px-4">
      <div className="w-32 h-32 relative">
        <img src={aiOrb} alt="AI Orb" className="w-full h-full object-contain drop-shadow-[0_0_30px_hsl(var(--primary)/0.5)]" />
      </div>
      <div className="text-center space-y-3">
        <h2 className="text-3xl font-bold text-foreground">Ready to Ask Something?</h2>
        <p className="text-muted-foreground max-w-md">
          DSCE HelpDesk AI can answer questions about admissions, courses, fees, and campus information.
        </p>
      </div>
      <div className="flex gap-3 mt-4">
        {["Admissions", "Courses", "Fees", "Campus"].map((tag) => (
          <span key={tag} className="px-4 py-1.5 rounded-full text-xs font-medium bg-primary/10 text-primary border border-primary/20">
            {tag}
          </span>
        ))}
      </div>
    </div>
  );
}
