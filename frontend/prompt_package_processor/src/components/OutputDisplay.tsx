import { Copy, Check, AlertTriangle, Scale, FileText, Clock3 } from 'lucide-react';
import { useState } from 'react';
import type { JudgeCaseResponse } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

interface OutputDisplayProps {
  result: JudgeCaseResponse | null;
  error: string | null;
  isLoading?: boolean;
}

const getSimilarCaseTitle = (item: unknown, index: number) => {
  if (typeof item === 'string') return `Similar case ${index + 1}`;
  if (item && typeof item === 'object') {
    const record = item as Record<string, unknown>;
    const candidate = record.case_name ?? record.title ?? record.name ?? record.citation;
    if (typeof candidate === 'string' && candidate.trim()) {
      return candidate.trim();
    }
  }
  return `Similar case ${index + 1}`;
};

const getRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

const asString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value.trim() : null;

const asNumber = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

const asStringArray = (value: unknown): string[] => {
  if (!Array.isArray(value)) return [];
  return value.map((item) => asString(item)).filter((item): item is string => Boolean(item));
};

const renderMetadataLine = (label: string, value: string | number | null) => {
  if (value == null) return null;
  return (
    <p className="text-xs text-muted-foreground">
      <span className="font-medium text-foreground">{label}:</span> {value}
    </p>
  );
};

const renderTagList = (label: string, values: string[]) => {
  if (values.length === 0) return null;
  return (
    <p className="text-xs text-muted-foreground">
      <span className="font-medium text-foreground">{label}:</span> {values.join(', ')}
    </p>
  );
};

/** Strip TOP_K_RETRIEVED_CASES and INSTRUCTIONS block so we don't show prompt/retrieval boilerplate. */
const stripRetrievalBlock = (text: string): string => {
  const startMarker = '===TOP_K_RETRIEVED_CASES===';
  const idx = text.indexOf(startMarker);
  if (idx === -1) return text;
  const afterBlock = text.slice(idx);
  const lines = afterBlock.split('\n');
  let endLineIndex = 0;
  const instructionsMarker = '===INSTRUCTIONS_FOR_USING_TOP_K_CASES===';
  const groundingLine = '4) Grounding the final outcome';
  let inInstructions = false;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes(instructionsMarker)) inInstructions = true;
    if (inInstructions && lines[i].includes(groundingLine)) {
      endLineIndex = i + 1;
      break;
    }
  }
  const blockEnd =
    endLineIndex > 0 ? idx + lines.slice(0, endLineIndex).join('\n').length : text.length;
  const before = text.slice(0, idx).trimEnd();
  const after = text.slice(blockEnd).trimStart();
  return [before, after].filter(Boolean).join('\n\n');
};

const getCopyText = (result: JudgeCaseResponse) => {
  const parts = [
    result.case_summary ? `Case Summary\n${stripRetrievalBlock(result.case_summary)}` : '',
    result.case_decision ? `Case Decision\n${stripRetrievalBlock(result.case_decision)}` : '',
    result.oral_argument_prediction
      ? `Oral Argument Recommendation\n${result.oral_argument_prediction}`
      : '',
    result.oral_argument_summary
      ? `Oral Argument Justification\n${result.oral_argument_summary}`
      : '',
  ].filter(Boolean);
  const rawFallback = result.raw ? stripRetrievalBlock(result.raw) : '';
  return parts.join('\n\n') || rawFallback || '';
};

export const OutputDisplay = ({ result, error, isLoading }: OutputDisplayProps) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (text: string) => {
    await navigator.clipboard.writeText(text);
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

  if (error) {
    return (
      <Alert variant="destructive" className="fade-in">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Unable to analyze files</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!result) {
    return (
      <div className="output-area border border-border p-6 min-h-[200px] flex items-center justify-center">
        <p className="text-muted-foreground text-center">
          Judge response will appear here after analysis.
        </p>
      </div>
    );
  }

  const similarCases = Array.isArray(result.similar_cases) ? result.similar_cases : [];
  const hasStructuredResponse = Boolean(result.case_summary || result.case_decision);
  const hasOralArgumentData = Boolean(
    result.oral_argument_prediction ||
      result.oral_argument_summary ||
      result.oral_argument_raw ||
      result.oral_argument_error,
  );

  return (
    <div className="space-y-4 fade-in">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between gap-3">
            <CardTitle className="text-lg">Analysis Overview</CardTitle>
            <button
              onClick={() => handleCopy(getCopyText(result))}
              className="p-2 rounded-md hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
              title="Copy primary analysis"
            >
              {copied ? <Check className="w-4 h-4 text-primary" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="flex flex-wrap gap-2">
            {typeof result.model === 'string' && <Badge variant="secondary">Model: {result.model}</Badge>}
            {typeof result.num_documents === 'number' && (
              <Badge variant="outline">Documents: {result.num_documents}</Badge>
            )}
            {typeof result.retrieval_k === 'number' && (
              <Badge variant="outline">Retrieval top-k: {result.retrieval_k}</Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {result.retrieval_error && (
        <Alert className="fade-in">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Retrieval warning</AlertTitle>
          <AlertDescription>{result.retrieval_error}</AlertDescription>
        </Alert>
      )}

      {result.case_summary && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="w-4 h-4 text-primary" />
              Case Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm leading-6 whitespace-pre-wrap">
              {stripRetrievalBlock(result.case_summary)}
            </p>
          </CardContent>
        </Card>
      )}

      {result.case_decision && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Scale className="w-4 h-4 text-primary" />
              Case Decision
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm leading-6 whitespace-pre-wrap">
              {stripRetrievalBlock(result.case_decision)}
            </p>
          </CardContent>
        </Card>
      )}

      {hasOralArgumentData && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Clock3 className="w-4 h-4 text-primary" />
              Oral Argument Prediction
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {result.oral_argument_prediction ? (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium text-foreground">Predicted allocation:</span>
                <Badge variant="secondary">{result.oral_argument_prediction}</Badge>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No oral argument allocation was parsed from the model response.
              </p>
            )}

            {result.oral_argument_summary && (
              <p className="text-sm leading-6 whitespace-pre-wrap">{result.oral_argument_summary}</p>
            )}

            {result.oral_argument_error && (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Oral argument prediction warning</AlertTitle>
                <AlertDescription>{result.oral_argument_error}</AlertDescription>
              </Alert>
            )}

            {!result.oral_argument_prediction && !result.oral_argument_summary && result.oral_argument_raw && (
              <details className="rounded-md border p-2">
                <summary className="cursor-pointer text-sm font-medium text-foreground">
                  View raw oral argument output
                </summary>
                <pre className="mt-2 text-sm whitespace-pre-wrap break-words max-h-[320px] overflow-auto">
                  {result.oral_argument_raw}
                </pre>
              </details>
            )}
          </CardContent>
        </Card>
      )}

      {similarCases.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Similar Cases</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {similarCases.map((item, index) => (
              <div key={`similar-case-${index}`} className="rounded-lg border bg-muted/30 p-3">
                <p className="text-sm font-medium">{getSimilarCaseTitle(item, index)}</p>
                {(() => {
                  if (typeof item === 'string') {
                    return <p className="mt-2 text-sm leading-6 whitespace-pre-wrap">{item}</p>;
                  }

                  const record = getRecord(item);
                  if (!record) {
                    return <p className="mt-2 text-sm text-muted-foreground">No case details available.</p>;
                  }

                  const caseId = asString(record.case_id);
                  const caseType = asString(record.case_type);
                  const proceduralPosture = asString(record.procedural_posture);
                  const score = asNumber(record.score);
                  const summary = asString(record.summary);
                  const issueTags = asStringArray(record.issue_tags);
                  const statuteTags = asStringArray(record.statute_tags);
                  const doctrineTags = asStringArray(record.doctrine_tags);

                  return (
                    <div className="mt-2 space-y-2">
                      <div className="space-y-1">
                        {renderMetadataLine('Case ID', caseId)}
                        {renderMetadataLine('Case type', caseType)}
                        {renderMetadataLine('Similarity score', score != null ? score.toFixed(4) : null)}
                        {renderMetadataLine('Procedural posture', proceduralPosture)}
                      </div>

                      {(issueTags.length > 0 || statuteTags.length > 0 || doctrineTags.length > 0) && (
                        <div className="space-y-1">
                          {renderTagList('Issue tags', issueTags)}
                          {renderTagList('Statute tags', statuteTags)}
                          {renderTagList('Doctrine tags', doctrineTags)}
                        </div>
                      )}

                      {summary && <p className="text-sm leading-6 whitespace-pre-wrap">{summary}</p>}
                    </div>
                  );
                })()}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {!hasStructuredResponse && result.raw && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Raw Model Output</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-sm whitespace-pre-wrap break-words max-h-[420px] overflow-auto">
              {stripRetrievalBlock(result.raw)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
