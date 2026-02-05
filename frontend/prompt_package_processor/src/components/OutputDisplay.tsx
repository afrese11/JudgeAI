import { Copy, Check } from 'lucide-react';
import { useState } from 'react';

interface OutputDisplayProps {
  output: string;
  isLoading?: boolean;
}

export const OutputDisplay = ({ output, isLoading }: OutputDisplayProps) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(output);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isLoading) {
    return (
      <div className="output-area border border-border p-6 min-h-[200px] flex items-center justify-center">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse [animation-delay:150ms]" />
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse [animation-delay:300ms]" />
          <span className="text-muted-foreground ml-2">Processing files...</span>
        </div>
      </div>
    );
  }

  if (!output) {
    return (
      <div className="output-area border border-border p-6 min-h-[200px] flex items-center justify-center">
        <p className="text-muted-foreground text-center">
          Output will appear here after processing
        </p>
      </div>
    );
  }

  return (
    <div className="output-area border border-border min-h-[200px] relative fade-in">
      <div className="absolute top-3 right-3">
        <button
          onClick={handleCopy}
          className="p-2 rounded-md hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
          title="Copy to clipboard"
        >
          {copied ? <Check className="w-4 h-4 text-primary" /> : <Copy className="w-4 h-4" />}
        </button>
      </div>
      <pre className="p-6 pr-12 whitespace-pre-wrap break-words text-foreground overflow-auto max-h-[400px]">
        {output}
      </pre>
    </div>
  );
};
