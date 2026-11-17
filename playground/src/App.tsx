import { useEffect, useRef, useState } from 'react';
import { EmbeddingScene } from './three/embeddingScene';
import { loadModelSpec, type SelInfo } from './three/modelMap';

export default function App(): React.JSX.Element {
  const canvasRef: React.RefObject<HTMLCanvasElement | null> = useRef<HTMLCanvasElement | null>(null);
  const [sel, setSel] = useState<SelInfo | null>(null);

  useEffect(() => {
    let cancelled = false;
    let scene: EmbeddingScene | null = null;
    loadModelSpec().then((s) => {
      if (cancelled || !canvasRef.current) return;
      scene = new EmbeddingScene(canvasRef.current, s, setSel);
    });
    return () => {
      cancelled = true;
      scene?.dispose();
    };
  }, []);

  return (
    <>
      <canvas id="scene" ref={canvasRef} />
      <div id="hud">
        <div>名称 <span id="token">{sel === null ? '-' : sel.title}</span></div>
        <div>形状 <span className="dim">{sel === null ? '-' : sel.shape}</span></div>
        <div>类型 <span className="dim">{sel === null ? '-' : sel.dtype}</span></div>
      </div>
    </>
  );
}
