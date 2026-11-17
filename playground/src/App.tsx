import { useEffect, useRef, useState } from 'react';
import { EmbeddingScene, VOCAB_SIZE, HIDDEN_SIZE } from './three/embeddingScene';

export default function App(): React.JSX.Element {
  const canvasRef: React.RefObject<HTMLCanvasElement | null> = useRef<HTMLCanvasElement | null>(null);
  const [tokenId, setTokenId] = useState<number | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const scene: EmbeddingScene = new EmbeddingScene(canvasRef.current, setTokenId);
    return () => scene.dispose();
  }, []);

  return (
    <>
      <canvas id="scene" ref={canvasRef} />
      <div id="hud">
        <div><b>embed_tokens</b> <span className="dim">[{VOCAB_SIZE}, {HIDDEN_SIZE}]</span> BF16 · 约 272 MB</div>
        <div>采样显示: 64 tokens × 64 dims（柱高/颜色 = 权重值）</div>
        <div>当前查表 token id ≈ <span id="token">{tokenId === null ? '-' : tokenId}</span></div>
      </div>
      <div id="help">左键旋转 · 右键平移 · 滚轮缩放</div>
    </>
  );
}
