// src/pages/Index.tsx
import { useState } from 'react';
import { FileDropZone } from '@/components/FileDropZone';
import { OutputDisplay } from '@/components/OutputDisplay';
import { Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Index = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [output, setOutput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleProcess = async () => {
    if (files.length === 0) return;

    setIsLoading(true);
    setOutput('');

    try {
      const form = new FormData();
      files.forEach((f) => form.append('files', f));
      form.append('redact', 'false'); // later: hook this to a toggle UI if you want

      const res = await fetch('http://127.0.0.1:8000/api/judge', {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed: ${res.status}`);
      }

      const data: {
        raw?: string;
        case_summary?: string | null;
        case_decision?: string | null;
        error?: string;
      } = await res.json();

      if (data.error) throw new Error(data.error);

      // Display the parsed output if possible; fallback to raw
      const pretty =
        data.case_summary && data.case_decision
          ? [
              '===CASE SUMMARY===',
              data.case_summary,
              '',
              '===CASE DECISION===',
              data.case_decision,
            ].join('\n')
          : data.raw || 'No output returned.';

      setOutput(pretty);
    } catch (err: any) {
      setOutput(`Error: ${err?.message ?? String(err)}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container max-w-3xl py-12 px-4">
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold text-foreground mb-2">JudgeAI</h1>
          <p className="text-muted-foreground">
            Upload legal documents for AI analysis.
          </p>
        </div>

        <div className="space-y-6">
          <FileDropZone files={files} onFilesChange={setFiles} />

          <Button
            onClick={handleProcess}
            disabled={files.length === 0 || isLoading}
            className="w-full h-12 text-base font-medium glow-button"
            size="lg"
          >
            <Sparkles className="w-5 h-5 mr-2" />
            {isLoading ? 'Processing...' : 'Go'}
          </Button>

          <OutputDisplay output={output} isLoading={isLoading} />
        </div>
      </div>
    </div>
  );
};

export default Index;
