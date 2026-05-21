import { useState, useCallback } from 'react';

let nextId = 1;

export function useToasts() {
  const [toasts, setToasts] = useState([]);

  const push = useCallback((message, kind = 'success', duration = 3500) => {
    const id = nextId++;
    setToasts((t) => [...t, { id, message, kind }]);
    setTimeout(() => {
      setToasts((t) => t.filter((x) => x.id !== id));
    }, duration);
  }, []);

  const dismiss = useCallback((id) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  return { toasts, push, dismiss };
}
