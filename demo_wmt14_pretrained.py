# ============================================================================
# WMT14 with Pre-trained Tokenizers (No Training Required!)
# ============================================================================

import torch
import torch.nn as nn
from util.tokenizer_with_progress import BPETokenizer
from util.data_loader import DataLoader
from models.model.transformer import Transformer
import time
import sys
import argparse
import pickle
import os
from pathlib import Path
import json

def print_section(title):
    """섹션 헤더 출력"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")

# ============================================================================
# 사전 학습된 토크나이저 래퍼 클래스
# ============================================================================

class PretrainedBPETokenizer:
    """
    사전 학습된 GPT-2 BPE 토크나이저를 사용하는 래퍼 클래스
    - train() 호출 불필요 (이미 학습 완료)
    - 50,257개 vocab + merges 규칙 포함
    - 1초 만에 로드 완료
    """
    def __init__(self, model_name="gpt2", vocab_size=None):
        """
        Args:
            model_name: 사용할 사전 학습 모델 (기본: gpt2)
                       - "gpt2": 50K vocab (영어 최적화)
                       - "xlm-roberta-base": 250K vocab (다국어)
                       - "bert-base-uncased": 30K vocab (영어)
            vocab_size: 사용되지 않음 (호환성 유지용)
        """
        print(f"Loading pre-trained tokenizer: {model_name}...")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.vocab = list(self.tokenizer.get_vocab().keys())
        
        # 기존 코드 호환성을 위한 속성
        self.merges = {}  # 실제로는 tokenizer 내부에 있음
        
        elapsed = time.time() - start_time
        print(f"✓ Tokenizer loaded in {elapsed:.2f}s")
        print(f"  Model: {model_name}")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Type: BPE (Byte-Pair Encoding)")
    
    def train(self, corpus):
        """
        학습 불필요 - 이미 사전 학습 완료
        호환성을 위해 함수는 남겨두되, 즉시 리턴
        """
        print(f"\n{'='*80}")
        print(f"Pre-trained Tokenizer - No Training Required!")
        print(f"{'='*80}")
        print(f"✓ Skipping training (already trained on billions of tokens)")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Ready to use immediately!")
        print(f"{'='*80}\n")
        # 아무것도 하지 않음 - 이미 학습 완료!
    
    def tokenize(self, text):
        """
        텍스트를 토큰으로 변환
        """
        # HuggingFace tokenizer 사용
        return self.tokenizer.tokenize(text.lower())
    
    def encode(self, text):
        """
        텍스트를 토큰 ID로 변환
        """
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, ids):
        """
        토큰 ID를 텍스트로 변환
        """
        return self.tokenizer.decode(ids)


class PretrainedWordPieceTokenizer:
    """
    사전 학습된 BERT WordPiece 토크나이저를 사용하는 래퍼 클래스
    """
    def __init__(self, model_name="bert-base-uncased", vocab_size=None):
        """
        Args:
            model_name: 사용할 사전 학습 모델
                       - "bert-base-uncased": 30K vocab (영어, 소문자)
                       - "bert-base-cased": 29K vocab (영어, 대소문자)
                       - "bert-base-multilingual-cased": 119K vocab (다국어)
        """
        print(f"Loading pre-trained tokenizer: {model_name}...")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.vocab = list(self.tokenizer.get_vocab().keys())
        
        elapsed = time.time() - start_time
        print(f"✓ Tokenizer loaded in {elapsed:.2f}s")
        print(f"  Model: {model_name}")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Type: WordPiece")
    
    def train(self, corpus):
        """
        학습 불필요 - 이미 사전 학습 완료
        """
        print(f"\n{'='*80}")
        print(f"Pre-trained Tokenizer - No Training Required!")
        print(f"{'='*80}")
        print(f"✓ Skipping training (already trained on billions of tokens)")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Ready to use immediately!")
        print(f"{'='*80}\n")
    
    def tokenize(self, text):
        """
        텍스트를 토큰으로 변환
        """
        return self.tokenizer.tokenize(text.lower())


# ============================================================================
# 저장/로드 함수들
# ============================================================================

def save_tokenizer(tokenizer, filepath):
    """토크나이저를 파일로 저장"""
    print(f"Saving tokenizer to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 사전 학습 토크나이저는 모델 이름만 저장
    tokenizer_info = {
        'type': 'pretrained',
        'model_name': getattr(tokenizer.tokenizer, 'name_or_path', 'gpt2'),
        'vocab_size': tokenizer.vocab_size
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer_info, f)
    
    print(f"✓ Tokenizer info saved (model: {tokenizer_info['model_name']})")

def load_tokenizer(filepath, tokenizer_class=PretrainedBPETokenizer):
    """토크나이저를 파일에서 로드"""
    print(f"Loading tokenizer from {filepath}...")
    
    with open(filepath, 'rb') as f:
        tokenizer_info = pickle.load(f)
    
    if tokenizer_info['type'] == 'pretrained':
        # 사전 학습 토크나이저 재로드
        tokenizer = tokenizer_class(model_name=tokenizer_info['model_name'])
        print(f"✓ Pre-trained tokenizer loaded")
    else:
        # 구버전 호환성
        tokenizer = pickle.load(open(filepath, 'rb'))
    
    return tokenizer

def save_vocab(loader, filepath):
    """어휘사전을 파일로 저장"""
    print(f"Saving vocabulary to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    vocab_data = {
        'source_stoi': loader.source.vocab.stoi,
        'source_itos': loader.source.vocab.itos,
        'target_stoi': loader.target.vocab.stoi,
        'target_itos': loader.target.vocab.itos,
        'shared': loader.source.vocab is loader.target.vocab
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(vocab_data, f)
    
    print(f"✓ Vocabulary saved")

def load_vocab(loader, filepath):
    """어휘사전을 파일에서 로드"""
    print(f"Loading vocabulary from {filepath}...")
    
    with open(filepath, 'rb') as f:
        vocab_data = pickle.load(f)
    
    # vocab 객체 생성
    source_vocab_obj = type('obj', (object,), {
        'stoi': vocab_data['source_stoi'],
        'itos': vocab_data['source_itos'],
        '__len__': lambda self: len(vocab_data['source_stoi'])
    })()
    
    if vocab_data['shared']:
        loader.source.vocab = source_vocab_obj
        loader.target.vocab = source_vocab_obj
    else:
        target_vocab_obj = type('obj', (object,), {
            'stoi': vocab_data['target_stoi'],
            'itos': vocab_data['target_itos'],
            '__len__': lambda self: len(vocab_data['target_stoi'])
        })()
        
        loader.source.vocab = source_vocab_obj
        loader.target.vocab = target_vocab_obj
    
    print(f"✓ Vocabulary loaded")

def save_training_artifacts(tokenizer_en, tokenizer_de, loader, save_dir):
    """모든 학습 결과물을 저장"""
    print_section("저장 중...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_tokenizer(tokenizer_en, save_dir / 'tokenizer_en.pkl')
    save_tokenizer(tokenizer_de, save_dir / 'tokenizer_de.pkl')
    save_vocab(loader, save_dir / 'vocab.pkl')
    
    metadata = {
        'vocab_size': len(loader.source.vocab),
        'shared_vocab': loader.source.vocab is loader.target.vocab,
        'src_pad_idx': loader.source.vocab.stoi['<pad>'],
        'trg_pad_idx': loader.target.vocab.stoi['<pad>'],
        'trg_sos_idx': loader.target.vocab.stoi['<sos>'],
        'tokenizer_type': 'pretrained',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ All artifacts saved to: {save_dir}")

def load_training_artifacts(loader, load_dir):
    """저장된 학습 결과물을 로드"""
    print_section("기존 결과물 로드 중...")
    
    load_dir = Path(load_dir)
    
    # 메타데이터 로드
    with open(load_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Metadata:")
    print(f"  Created: {metadata['timestamp']}")
    print(f"  Vocab size: {metadata['vocab_size']:,}")
    print(f"  Tokenizer type: {metadata.get('tokenizer_type', 'custom')}")
    
    # 토크나이저 로드
    print()
    tokenizer_en = load_tokenizer(load_dir / 'tokenizer_en.pkl', PretrainedBPETokenizer)
    tokenizer_de = load_tokenizer(load_dir / 'tokenizer_de.pkl', PretrainedBPETokenizer)
    
    # 어휘사전 로드
    print()
    load_vocab(loader, load_dir / 'vocab.pkl')
    
    print(f"\n✓ All artifacts loaded from: {load_dir}")
    
    return tokenizer_en, tokenizer_de, metadata

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='WMT14 Transformer with Pre-trained Tokenizers')
    
    parser.add_argument(
        '--load_dir',
        type=str,
        default=None,
        help='Directory to load pre-built vocabulary (skip vocab building)'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='artifacts_pretrained',
        help='Directory to save vocabulary (default: artifacts_pretrained)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=25000,
        help='Maximum tokens per batch (default: 25000)'
    )
    
    parser.add_argument(
        '--tokenizer_en',
        type=str,
        default='gpt2',
        help='English tokenizer model (default: gpt2)'
    )
    
    parser.add_argument(
        '--tokenizer_de',
        type=str,
        default='gpt2',
        help='German tokenizer model (default: gpt2)'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Directory to save model checkpoints (default: checkpoints)'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*80)
    print(" "*15 + "WMT14 Transformer with Pre-trained Tokenizers")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Load directory: {args.load_dir if args.load_dir else 'None (build vocab from scratch)'}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Max tokens per batch: {args.max_tokens:,}")
    print(f"  English tokenizer: {args.tokenizer_en}")
    print(f"  German tokenizer: {args.tokenizer_de}")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    
    # ------------------------------------------------------------------------
    # 1. 데이터셋 로드
    # ------------------------------------------------------------------------
    print_section("1. 데이터셋 로드")
    
    loader = DataLoader(
        ext=('.en', '.de'),
        tokenize_en=None,
        tokenize_de=None,
        init_token='<sos>',
        eos_token='<eos>'
    )
    
    print("Loading WMT14 dataset...")
    start_time = time.time()
    
    train, valid, test = loader.make_dataset(dataset_name='wmt14')
    
    elapsed = time.time() - start_time
    print(f"\n✓ Dataset loaded in {elapsed:.1f}s")
    print(f"  Train: {len(train):,} sentence pairs")
    print(f"  Valid: {len(valid):,} sentence pairs")
    print(f"  Test:  {len(test):,} sentence pairs")
    
    # ------------------------------------------------------------------------
    # 2. 사전 학습 토크나이저 로드 (즉시!)
    # ------------------------------------------------------------------------
    if args.load_dir:
        # 기존 결과물 로드
        tokenizer_en, tokenizer_de, metadata = load_training_artifacts(loader, args.load_dir)
        loader.tokenize_en = tokenizer_en.tokenize
        loader.tokenize_de = tokenizer_de.tokenize
        
    else:
        print_section("2. 사전 학습 토크나이저 로드")
        
        print("=" * 80)
        print("🚀 Using Pre-trained Tokenizers - No 42-hour Training!")
        print("=" * 80)
        print()
        
        # 영어 토크나이저 로드 (1초!)
        print("English Tokenizer:")
        tokenizer_en = PretrainedBPETokenizer(model_name=args.tokenizer_en)
        
        print()
        
        # 독일어 토크나이저 로드 (1초!)
        print("German Tokenizer:")
        tokenizer_de = PretrainedBPETokenizer(model_name=args.tokenizer_de)
        
        print()
        print("=" * 80)
        print("✓ Both tokenizers ready! (Total time: ~2 seconds)")
        print("  ⏱️  Time saved: ~84 hours (42h EN + 42h DE)")
        print("=" * 80)
        
        # DataLoader에 토크나이저 설정
        loader.tokenize_en = tokenizer_en.tokenize
        loader.tokenize_de = tokenizer_de.tokenize
        
        # ------------------------------------------------------------------------
        # 3. 어휘 사전 구축
        # ------------------------------------------------------------------------
        print_section("3. 어휘 사전 구축")
        
        print("Building vocabulary from tokenized data...")
        vocab_start = time.time()
        
        # 사전 학습 토크나이저의 vocab 크기 사용
        effective_vocab_size = min(tokenizer_en.vocab_size, tokenizer_de.vocab_size)
        
        loader.build_vocab(
            train_data=train,
            min_freq=2,
            max_vocab_size=effective_vocab_size,
            shared_vocab=True
        )
        
        print(f"\n✓ Vocabulary built in {time.time() - vocab_start:.1f}s")
        
        # ------------------------------------------------------------------------
        # 4. 결과물 저장
        # ------------------------------------------------------------------------
        save_training_artifacts(tokenizer_en, tokenizer_de, loader, args.save_dir)
    
    # ------------------------------------------------------------------------
    # 어휘 정보 출력
    # ------------------------------------------------------------------------
    src_pad_idx = loader.source.vocab.stoi['<pad>']
    trg_pad_idx = loader.target.vocab.stoi['<pad>']
    trg_sos_idx = loader.target.vocab.stoi['<sos>']
    
    enc_voc_size = len(loader.source.vocab)
    dec_voc_size = len(loader.target.vocab)
    
    print_section("어휘 정보")
    print(f"Vocabulary Statistics:")
    print(f"  Source vocab size: {enc_voc_size:,}")
    print(f"  Target vocab size: {dec_voc_size:,}")
    print(f"  Shared: {loader.source.vocab is loader.target.vocab}")
    print(f"\nSpecial Tokens:")
    print(f"  <pad>: {src_pad_idx}")
    print(f"  <unk>: {loader.source.vocab.stoi['<unk>']}")
    print(f"  <sos>: {trg_sos_idx}")
    print(f"  <eos>: {loader.source.vocab.stoi['<eos>']}")
    
    # ------------------------------------------------------------------------
    # 5. Iterator 생성
    # ------------------------------------------------------------------------
    print_section("5. 데이터 Iterator 생성")
    
    print("Creating data iterators...")
    iter_start = time.time()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_iter, valid_iter, test_iter = loader.make_iter(
        train, valid, test,
        max_tokens=args.max_tokens,
        device=device,
        num_workers=4
    )
    
    print(f"✓ Iterators created in {time.time() - iter_start:.1f}s")
    print(f"\nIterator Statistics:")
    print(f"  Train batches: {len(train_iter):,}")
    print(f"  Valid batches: {len(valid_iter):,}")
    
    # ------------------------------------------------------------------------
    # 6. 모델 초기화
    # ------------------------------------------------------------------------
    print_section("6. 모델 초기화")
    
    print(f"Using device: {device}")
    print("\nInitializing Transformer model...")
    
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        d_model=512,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=512,
        ffn_hidden=2048,
        n_head=8,
        n_layers=6,
        drop_prob=0.1,
        device=device
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    
    # ------------------------------------------------------------------------
    # 7. Optimizer 및 Scheduler
    # ------------------------------------------------------------------------
    print_section("7. Optimizer 및 Scheduler")
    
    from torch.optim import Adam
    from torch.optim.lr_scheduler import LambdaLR
    
    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    
    def lr_schedule(step):
        d_model = 512
        warmup_steps = 4000
        return (d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)
    
    print("✓ Training setup complete")
    
    # ------------------------------------------------------------------------
    # 8. 학습 루프
    # ------------------------------------------------------------------------
    print_section("8. 학습 시작")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Ready to train for {args.epochs} epochs")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    input("\nPress Enter to start training...")
    
    training_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")
        
        for batch_idx, batch in enumerate(train_iter):
            src, trg = batch.src, batch.trg
            
            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            
            loss = criterion(
                output.contiguous().view(-1, output.shape[-1]),
                trg[:, 1:].contiguous().view(-1)
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                elapsed = time.time() - epoch_start
                progress = (batch_idx + 1) / len(train_iter) * 100
                avg_loss = epoch_loss / (batch_idx + 1)
                
                print(f"  Batch {batch_idx + 1:,}/{len(train_iter):,} ({progress:.1f}%) | "
                      f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f})", end='\r')
        
        avg_train_loss = epoch_loss / len(train_iter)
        print(f"  Epoch {epoch + 1} complete | Train Loss: {avg_train_loss:.4f}                    ")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_iter:
                output = model(batch.src, batch.trg[:, :-1])
                loss = criterion(
                    output.contiguous().view(-1, output.shape[-1]),
                    batch.trg[:, 1:].contiguous().view(-1)
                )
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(valid_iter)
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        
        # 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    print_section("학습 완료!")
    print(f"Total time: {(time.time() - training_start)/3600:.2f} hours")

if __name__ == '__main__':
    main()
