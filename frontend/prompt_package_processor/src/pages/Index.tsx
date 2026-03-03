import { useMemo, useState } from 'react';
import { FileDropZone } from '@/components/FileDropZone';
import { OutputDisplay } from '@/components/OutputDisplay';
import { Sparkles, ShieldCheck } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { submitJudgeCase, type JudgeCaseResponse } from '@/lib/api';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

const Index = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<JudgeCaseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [redact, setRedact] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const apiBase = useMemo(() => {
    const envBase = import.meta.env.VITE_API_BASE_URL as string | undefined;
    if (envBase && envBase.trim()) {
      return envBase.trim().replace(/\/+$/, '');
    }
    // Local dev fallback when Vite proxy is unavailable.
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://127.0.0.1:8000';
    }
    return '';
  }, []);

  const handleValidationError = (message: string) => {
    setError(message);
  };

  const handleAnalyze = async () => {
    if (files.length === 0) {
      setError('Add at least one PDF brief before running analysis.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await submitJudgeCase({
        files,
        redact,
        apiBase,
      });
      setResult(response);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
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
            Drop related case briefs and get a concise structured analysis.
          </p>
        </div>

        <div className="space-y-6">
          <FileDropZone
            files={files}
            onFilesChange={setFiles}
            onValidationError={handleValidationError}
            disabled={isLoading}
          />

          <Alert>
            <ShieldCheck className="h-4 w-4" />
            <AlertTitle>Backend note</AlertTitle>
            <AlertDescription>
              This client sends PDFs directly to <code>/api/judge</code>. If your backend has passcode
              protection enabled, requests will be rejected until passcode support is reintroduced.
            </AlertDescription>
          </Alert>

          <div className="flex items-center justify-between rounded-lg border bg-card px-4 py-3">
            <label htmlFor="redact-toggle" className="text-sm font-medium text-foreground">
              Redact sensitive details before analysis
            </label>
            <input
              id="redact-toggle"
              type="checkbox"
              checked={redact}
              disabled={isLoading}
              onChange={(event) => setRedact(event.target.checked)}
              className="h-4 w-4 accent-primary"
            />
          </div>

          <Button
            onClick={handleAnalyze}
            disabled={files.length === 0 || isLoading}
            className="w-full h-12 text-base font-medium glow-button"
            size="lg"
          >
            <Sparkles className="w-5 h-5 mr-2" />
            {isLoading ? 'Analyzing briefs...' : 'Analyze Case Briefs'}
          </Button>

          <OutputDisplay result={result} error={error} isLoading={isLoading} />
        </div>
      </div>
    </div>
  );
};

export default Index;
