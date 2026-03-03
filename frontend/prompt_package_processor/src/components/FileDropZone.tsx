import { useCallback, useState } from 'react';
import { Upload, X, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface FileDropZoneProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
  disabled?: boolean;
  onValidationError?: (message: string) => void;
}

const formatFileSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const isPdf = (file: File) =>
  file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');

const uniqueFiles = (existing: File[], incoming: File[]) => {
  const seen = new Set(existing.map((f) => `${f.name}-${f.size}-${f.lastModified}`));
  const merged = [...existing];

  for (const file of incoming) {
    const key = `${file.name}-${file.size}-${file.lastModified}`;
    if (!seen.has(key)) {
      seen.add(key);
      merged.push(file);
    }
  }

  return merged;
};

export const FileDropZone = ({
  files,
  onFilesChange,
  disabled = false,
  onValidationError,
}: FileDropZoneProps) => {
  const [isDragActive, setIsDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragActive(true);
    }
  }, []);

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  }, []);

  const addFiles = useCallback((incomingFiles: File[]) => {
    if (incomingFiles.length === 0) return;

    const valid = incomingFiles.filter(isPdf);
    const rejected = incomingFiles.length - valid.length;
    if (rejected > 0) {
      onValidationError?.('Only PDF files are supported.');
    }

    if (valid.length > 0) {
      onFilesChange(uniqueFiles(files, valid));
    }
  }, [files, onFilesChange, onValidationError]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (disabled) return;

    const droppedFiles = Array.from(e.dataTransfer.files);
    addFiles(droppedFiles);
  }, [addFiles, disabled]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return;
    const selectedFiles = e.target.files ? Array.from(e.target.files) : [];
    addFiles(selectedFiles);
    e.currentTarget.value = '';
  }, [addFiles, disabled]);

  const removeFile = useCallback((index: number) => {
    onFilesChange(files.filter((_, i) => i !== index));
  }, [files, onFilesChange]);

  const clearAllFiles = useCallback(() => {
    onFilesChange([]);
  }, [onFilesChange]);

  return (
    <div className="space-y-4">
      <div
        className={`drop-zone p-8 ${disabled ? 'cursor-not-allowed opacity-80' : 'cursor-pointer'} ${isDragActive ? 'drop-zone-active' : ''}`}
        onDragEnter={handleDragIn}
        onDragLeave={handleDragOut}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => {
          if (!disabled) {
            document.getElementById('file-input')?.click();
          }
        }}
      >
        <input
          id="file-input"
          type="file"
          accept=".pdf,application/pdf"
          multiple
          disabled={disabled}
          className="hidden"
          onChange={handleFileInput}
        />
        <div className="flex flex-col items-center justify-center gap-4 py-8">
          <div className={`p-4 rounded-full bg-primary/10 transition-transform duration-300 ${isDragActive ? 'scale-110' : ''}`}>
            <Upload className="w-8 h-8 text-primary" />
          </div>
          <div className="text-center">
            <p className="text-lg font-medium text-foreground">
              {isDragActive ? 'Drop files here' : 'Drop files here or click to browse'}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              PDF only, multi-file upload supported
            </p>
          </div>
        </div>
      </div>

      {files.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm font-medium text-muted-foreground">
              {files.length} PDF file{files.length !== 1 ? 's' : ''} selected
            </p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={clearAllFiles}
              disabled={disabled}
            >
              Clear all
            </Button>
          </div>
          <div className="max-h-48 overflow-y-auto space-y-2 pr-2">
            {files.map((file, index) => (
              <div
                key={`${file.name}-${index}`}
                className="file-item p-3 flex items-center justify-between gap-3 fade-in border border-border"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="text-muted-foreground">
                    <FileText className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">
                      {file.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(file.size)}
                    </p>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(index);
                  }}
                  disabled={disabled}
                  className="p-1.5 rounded-md hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
