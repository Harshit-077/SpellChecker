import { Github, Heart } from "lucide-react";

const Footer = () => {
  return (
    <footer className="w-full py-8 px-4 md:px-8 border-t border-border">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>Built with</span>
          <Heart className="w-4 h-4 text-destructive fill-destructive" />
          <span>using PyTorch Seq2Seq</span>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-secondary rounded-lg">
            <span className="text-xs font-medium text-secondary-foreground">
              Model: seq2seq_best.pt
            </span>
          </div>
          <a
            href="#"
            className="flex items-center gap-2 px-3 py-1.5 bg-foreground/5 hover:bg-foreground/10 rounded-lg transition-colors duration-200"
          >
            <Github className="w-4 h-4 text-foreground" />
            <span className="text-xs font-medium text-foreground">Source</span>
          </a>
        </div>
      </div>
      
      <div className="max-w-6xl mx-auto mt-6 pt-6 border-t border-border/50">
        <p className="text-xs text-center text-muted-foreground">
          Note: This is a frontend demo. To use the actual model, connect to a Python backend running the seq2seq_best.pt checkpoint.
        </p>
      </div>
    </footer>
  );
};

export default Footer;
