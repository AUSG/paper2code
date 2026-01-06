#!/usr/bin/env python3
"""
Vast.ai 자동 인스턴스 선택 및 실행 스크립트 v3
- 콘솔과 완전히 동일한 API 요청
- image_tag, type, duration 등 모든 파라미터 포함
"""

import argparse
import requests
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List


# ============================================================
# 설정
# ============================================================

@dataclass
class Config:
    # Vast.ai API 키
    api_key: str = ""

    # Docker 이미지 설정 (vastai 기본 이미지 사용)
    docker_image: str = "vastai/pytorch:@vastai-automatic-tag"

    # 콘솔 스타일 이미지 태그 (자동 CUDA 버전 매칭)
    # None이면 image_tag 필터 사용 안함 (더 많은 결과)
    # "vastai/pytorch:@vastai-automatic-tag" 등 사용 가능
    image_tag: str = None

    # GitHub 설정 (on_start.sh에서 프로젝트 clone용)
    gh_token: str = ""
    gh_user: str = ""
    gh_repo: str = ""
    gh_branch: str = "main"

    # 인스턴스 요구사항
    max_price_per_hour: float = 10.0
    min_reliability: float = 0.9
    min_disk_space: float = 50.0
    min_inet_down: float = 500.0
    min_inet_up: float = 800.0
    num_gpus: int = 8
    min_duration: float = 604800  # 최소 7일 (초 단위)
    min_direct_ports: int = 0

    # 선호 GPU
    preferred_gpu: str = "RTX PRO 6000 S"

    # 실행 설정
    disk_space: float = 700.0

    # 검색 옵션
    include_unverified: bool = False
    offer_type: str = "ask"  # 콘솔 기본값: "ask"

    # 환경변수 (Docker 컨테이너에 전달)
    env_vars: dict = field(default_factory=dict)

    # 학습 설정
    extra_args: str = ""

    # 디버그 모드
    debug: bool = False


# ============================================================
# API 클라이언트
# ============================================================

class VastClient:
    BASE_URL = "https://console.vast.ai/api/v0"

    def __init__(self, api_key: str, debug: bool = False):
        self.api_key = api_key
        self.debug = debug

    def _log(self, msg: str):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def _log_request(self, method: str, url: str, payload: dict):
        if self.debug:
            print("\n" + "=" * 60)
            print(f"[DEBUG] {method} {url}")
            print("-" * 60)
            print("Payload:")
            print(json.dumps(payload, indent=2))
            print("=" * 60)

    def _log_response(self, response: requests.Response, data: dict = None):
        if self.debug:
            print(f"[DEBUG] Status: {response.status_code}")
            if data and "offers" in data:
                print(f"[DEBUG] Offers count: {len(data.get('offers', []))}")

    def search_offers_console_style(self, cfg: Config) -> list:
        """
        콘솔과 완전히 동일한 방식으로 검색
        """
        url = f"{self.BASE_URL}/bundles/"
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        # 콘솔과 동일한 페이로드 구성
        payload = {
            # 기본 필터
            "type": cfg.offer_type,  # "ask" (콘솔 기본)
            "rentable": {"eq": True},
            "resource_type": "gpu",
            
            # verified 필터
            "verified": {"eq": not cfg.include_unverified},
            
            # 신뢰도
            "reliability2": {"gte": cfg.min_reliability},
            
            # 디스크 및 스토리지
            "disk_space": {"gte": cfg.min_disk_space},
            "allocated_storage": cfg.disk_space,
            
            # 최소 대여 기간
            "duration": {"gte": cfg.min_duration},
            
            # 포트
            "direct_port_count": {"gte": cfg.min_direct_ports},
            
            # CPU 아키텍처
            "cpu_arch": {"in": ["arm64", "amd64"]},
            
            # 정렬
            "order": [["dlperf_per_dphtotal", "desc"]],
            "sort_option": {"0": ["dlperf_per_dphtotal", "desc"]},
            
            # 제한
            "limit": 1500,
        }
        
        # GPU 필터
        if cfg.preferred_gpu:
            payload["gpu_name"] = {"eq": cfg.preferred_gpu}
        
        # GPU 개수 필터
        if cfg.num_gpus:
            payload["num_gpus"] = {"eq": cfg.num_gpus}
        
        # 이미지 태그 필터 (핵심!)
        # 이걸 넣으면 해당 이미지를 지원하는 인스턴스만 반환
        if cfg.image_tag:
            payload["image_tag"] = {"eq": cfg.image_tag}
        
        params = {"api_key": self.api_key}
        
        self._log_request("POST", url, payload)
        
        response = requests.post(url, headers=headers, params=params, json=payload)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return []
        
        data = response.json()
        self._log_response(response, data)
        return data.get("offers", [])

    def search_offers_no_image_filter(self, cfg: Config) -> list:
        """
        image_tag 필터 없이 검색 (더 많은 결과)
        """
        url = f"{self.BASE_URL}/bundles/"
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        payload = {
            "type": cfg.offer_type,
            "rentable": {"eq": True},
            "resource_type": "gpu",
            "verified": {"eq": not cfg.include_unverified},
            "reliability2": {"gte": cfg.min_reliability},
            "disk_space": {"gte": cfg.min_disk_space},
            "allocated_storage": cfg.disk_space,
            "duration": {"gte": cfg.min_duration},
            "direct_port_count": {"gte": cfg.min_direct_ports},
            "cpu_arch": {"in": ["arm64", "amd64"]},
            "order": [["dlperf_per_dphtotal", "desc"]],
            "limit": 1500,
        }
        
        if cfg.preferred_gpu:
            payload["gpu_name"] = {"eq": cfg.preferred_gpu}
        
        if cfg.num_gpus:
            payload["num_gpus"] = {"eq": cfg.num_gpus}
        
        params = {"api_key": self.api_key}
        
        self._log_request("POST (no image filter)", url, payload)
        
        response = requests.post(url, headers=headers, params=params, json=payload)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            return []
        
        data = response.json()
        self._log_response(response, data)
        return data.get("offers", [])

    def create_instance(self, offer_id: int, image: str, disk: float,
                        onstart: str = "", env: dict = None,
                        image_login: str = "") -> dict:
        """인스턴스 생성"""
        url = f"{self.BASE_URL}/asks/{offer_id}/"
        params = {"api_key": self.api_key}
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        payload = {
            "client_id": "me",
            "image": image,
            "disk": disk,
            "onstart": onstart,
        }

        if env:
            payload["env"] = env

        if image_login:
            payload["image_login"] = image_login

        self._log_request("PUT", url, payload)

        response = requests.put(url, headers=headers, params=params, json=payload)
        if response.status_code != 200:
            print(f"API Error: {response.text}")
        response.raise_for_status()
        return response.json()

    def get_instances(self) -> list:
        """내 인스턴스 목록"""
        url = f"{self.BASE_URL}/instances/"
        headers = {"Accept": "application/json"}
        params = {"api_key": self.api_key, "owner": "me"}

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get("instances", [])


# ============================================================
# 인스턴스 선택
# ============================================================

def filter_offers(offers: list, cfg: Config) -> list:
    """추가 필터링 (API 결과에서)"""
    filtered = []

    for offer in offers:
        price = offer.get("dph_total", 999)
        inet_down = offer.get("inet_down", 0)
        inet_up = offer.get("inet_up", 0)

        if price > cfg.max_price_per_hour:
            continue
        if inet_down < cfg.min_inet_down:
            continue
        if inet_up < cfg.min_inet_up:
            continue
        if not offer.get("rentable", False):
            continue

        filtered.append(offer)

    return filtered


def score_offer(offer: dict, cfg: Config) -> float:
    """점수 계산"""
    score = 0.0
    price = offer.get("dph_total", 999)
    reliability = offer.get("reliability2", 0)
    dlperf = offer.get("dlperf_per_dphtotal", 0)

    # 가격 점수
    price_score = max(0, 30 * (1 - price / cfg.max_price_per_hour))
    score += price_score

    # 신뢰도
    score += reliability * 5

    # DL 성능/$
    score += dlperf * 0.1

    if offer.get("verification") == "verified":
        score += 2

    return score


def select_best_offer(offers: list, cfg: Config) -> Optional[dict]:
    """최적 인스턴스 선택"""
    valid_offers = filter_offers(offers, cfg)
    
    if not valid_offers:
        return None

    scored = [(offer, score_offer(offer, cfg)) for offer in valid_offers]
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[0][0]


# ============================================================
# 출력
# ============================================================

def print_offer_info(offer: dict):
    """인스턴스 정보"""
    print("\n" + "=" * 50)
    print("Selected Instance")
    print("=" * 50)
    print(f"  ID: {offer.get('id')}")
    print(f"  GPU: {offer.get('gpu_name')} x {offer.get('num_gpus', 1)}")
    print(f"  VRAM: {offer.get('gpu_ram', 0) / 1024:.1f} GB")
    print(f"  Price: ${offer.get('dph_total', 0):.3f}/hour")
    print(f"  Reliability: {offer.get('reliability2', 0) * 100:.1f}%")
    print(f"  DLPerf/$: {offer.get('dlperf_per_dphtotal', 0):.2f}")
    print(f"  Download: {offer.get('inet_down', 0):.0f} Mbps")
    print(f"  Upload: {offer.get('inet_up', 0):.0f} Mbps")
    print(f"  Disk: {offer.get('disk_space', 0):.0f} GB")
    print(f"  Location: {offer.get('geolocation', 'Unknown')}")
    print(f"  CUDA Max: {offer.get('cuda_max_good', 'unknown')}")
    print(f"  Duration: {offer.get('duration', 0) / 86400:.1f} days")
    print("=" * 50)


def print_search_summary(all_offers: int, filtered_offers: int, cfg: Config):
    """검색 요약"""
    print("\n" + "-" * 50)
    print("Search Summary")
    print("-" * 50)
    print(f"  API results: {all_offers}")
    print(f"  After price/inet filter: {filtered_offers}")
    print(f"  Type: {cfg.offer_type}")
    print(f"  Image tag filter: {cfg.image_tag or 'None (disabled)'}")
    print(f"  Verified only: {not cfg.include_unverified}")
    print(f"  GPUs: {cfg.num_gpus} x {cfg.preferred_gpu}")
    print(f"  Max price: ${cfg.max_price_per_hour}/hr")
    print(f"  Min duration: {cfg.min_duration / 86400:.0f} days")
    print("-" * 50)


def build_env_vars(cfg: Config, instance_id: str = "") -> dict:
    """환경변수 - PROVISIONING_SCRIPT 포함"""
    env = {
        "WANDB_API_KEY": cfg.env_vars.get("wandb_key", ""),
        "HF_TOKEN": cfg.env_vars.get("hf_token", ""),
        "AWS_ACCESS_KEY_ID": cfg.env_vars.get("aws_key", ""),
        "AWS_SECRET_ACCESS_KEY": cfg.env_vars.get("aws_secret", ""),
        "AWS_DEFAULT_REGION": cfg.env_vars.get("aws_region", "ap-northeast-2"),
        "S3_DATA_PATH": cfg.env_vars.get("s3_path", ""),
        "VAST_API_KEY": cfg.api_key,
        "CONTAINER_ID": instance_id,
        # GitHub 설정
        "GH_TOKEN": cfg.gh_token,
        "GH_USER": cfg.gh_user,
        "GH_REPO": cfg.gh_repo,
        "BRANCH": cfg.gh_branch,
        # 핵심: provisioning script URL (entrypoint.sh가 Supervisor 시작 후 실행)
        "PROVISIONING_SCRIPT": "https://gist.githubusercontent.com/silverstar0727/de71fa128717fa7a9359b97aa09cd7f1/raw/d93e67294fa0b7c60ebb906904265d2784e5e93b/vastai_start.sh",
    }
    return {k: v for k, v in env.items() if v}


def generate_onstart_script() -> str:
    """on_start.sh 스크립트 내용 생성

    vast.ai base-image의 entrypoint.sh가 블로킹되므로,
    on_start.sh에서는 entrypoint.sh만 호출하고
    실제 프로비저닝은 PROVISIONING_SCRIPT 환경변수로 처리합니다.
    """
    return '''#!/bin/bash
# vast.ai 기본 entrypoint 호출 (이것만!)
# PROVISIONING_SCRIPT 환경변수가 설정되어 있으면
# entrypoint.sh가 Supervisor 시작 후 해당 스크립트를 실행합니다.
exec entrypoint.sh
'''


# ============================================================
# 메인 기능
# ============================================================

def compare_with_without_image_tag(cfg: Config):
    """image_tag 유무에 따른 결과 비교"""
    print("\n" + "=" * 60)
    print("Comparing with/without image_tag filter...")
    print("=" * 60)
    
    client = VastClient(cfg.api_key, debug=cfg.debug)
    
    # image_tag 없이
    print("\n[1] WITHOUT image_tag filter:")
    offers_no_tag = client.search_offers_no_image_filter(cfg)
    print(f"    -> {len(offers_no_tag)} offers")
    
    # image_tag 있이
    if cfg.image_tag:
        print(f"\n[2] WITH image_tag filter ({cfg.image_tag}):")
        offers_with_tag = client.search_offers_console_style(cfg)
        print(f"    -> {len(offers_with_tag)} offers")
    
    # GPU 분포
    if offers_no_tag:
        print("\n[GPU Distribution (no image_tag)]:")
        gpu_counts = {}
        for offer in offers_no_tag:
            gpu = offer.get("gpu_name", "unknown")
            num = offer.get("num_gpus", 1)
            key = f"{gpu} x{num}"
            gpu_counts[key] = gpu_counts.get(key, 0) + 1
        
        for gpu, count in sorted(gpu_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"    {gpu}: {count}")


def launch(cfg: Config, dry_run: bool = False, max_retries: int = 5) -> Optional[dict]:
    """메인 실행 (재시도 로직 포함)"""
    print("Searching for instances on Vast.ai...")
    print(f"  Type: {cfg.offer_type}")
    print(f"  Image tag: {cfg.image_tag or 'None'}")

    client = VastClient(cfg.api_key, debug=cfg.debug)

    # image_tag가 있으면 콘솔 스타일, 없으면 필터 없이
    if cfg.image_tag:
        offers = client.search_offers_console_style(cfg)
    else:
        offers = client.search_offers_no_image_filter(cfg)

    print(f"Found {len(offers)} offers from API")

    filtered = filter_offers(offers, cfg)
    print_search_summary(len(offers), len(filtered), cfg)

    if not filtered:
        print("\nNo matching instances found.")
        print("\nTry:")
        print("  --no-image-tag  (disable image compatibility filter)")
        print("  --include-unverified")

        if offers:
            print(f"\nSample offers ({len(offers)}):")
            for offer in offers[:10]:
                print(f"  {offer.get('gpu_name'):20s} x{offer.get('num_gpus', 1)} | "
                      f"${offer.get('dph_total', 0):.2f}/hr | "
                      f"duration: {offer.get('duration', 0)/86400:.0f}d")
        return None

    # 점수순으로 정렬
    scored = [(offer, score_offer(offer, cfg)) for offer in filtered]
    scored.sort(key=lambda x: x[1], reverse=True)

    if dry_run:
        best = scored[0][0]
        print_offer_info(best)
        print("\n[DRY RUN] Skipping instance creation")
        return best

    # on_start.sh 스크립트 생성
    onstart_cmd = generate_onstart_script()

    # 재시도 로직: 상위 인스턴스들을 순차적으로 시도
    for attempt, (offer, score) in enumerate(scored[:max_retries], 1):
        print(f"\n[Attempt {attempt}/{min(len(scored), max_retries)}]")
        print_offer_info(offer)

        env_vars = build_env_vars(cfg, str(offer["id"]))

        # 디버그: 전달되는 환경변수 확인 (값은 마스킹)
        print(f"[DEBUG] Env vars being passed to container:")
        for k, v in env_vars.items():
            if k in ["GH_TOKEN", "WANDB_API_KEY", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "VAST_API_KEY"]:
                print(f"  {k}: {'*' * min(len(v), 8)}... (len={len(v)})")
            else:
                print(f"  {k}: {v}")

        print(f"Creating instance...")
        print(f"  Image: {cfg.docker_image}")

        try:
            result = client.create_instance(
                offer_id=offer["id"],
                image=cfg.docker_image,
                disk=cfg.disk_space,
                onstart=onstart_cmd,
                env=env_vars,
            )
            instance_id = result.get('new_contract')
            print(f"\nInstance created! ID: {instance_id}")
            print(f"Monitor: https://cloud.vast.ai/instances/")
            return result
        except Exception as e:
            print(f"Creation failed: {e}")
            if attempt < min(len(scored), max_retries):
                print("Trying next available instance...")
            else:
                print("\nAll attempts failed. No more instances to try.")

    return None


def list_my_instances(cfg: Config):
    """인스턴스 목록"""
    client = VastClient(cfg.api_key)
    instances = client.get_instances()

    print(f"\nMy Instances ({len(instances)})")
    print("-" * 60)

    for inst in instances:
        status = inst.get("actual_status", "unknown")
        gpu = inst.get("gpu_name", "?")
        price = inst.get("dph_total", 0)
        print(f"  [{inst.get('id')}] {gpu} | ${price:.3f}/hr | {status}")


def search_only(cfg: Config):
    """검색만"""
    print("Searching...")
    
    client = VastClient(cfg.api_key, debug=cfg.debug)
    
    if cfg.image_tag:
        offers = client.search_offers_console_style(cfg)
    else:
        offers = client.search_offers_no_image_filter(cfg)
    
    filtered = filter_offers(offers, cfg)
    print_search_summary(len(offers), len(filtered), cfg)
    
    if not filtered:
        print("\nNo matching instances.")
        return
    
    print(f"\nMatching instances ({len(filtered)}):")
    print("-" * 90)
    
    scored = [(offer, score_offer(offer, cfg)) for offer in filtered]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    for offer, score in scored[:20]:
        print(f"  [{offer.get('id'):8d}] {offer.get('gpu_name'):20s} x{offer.get('num_gpus', 1)} | "
              f"${offer.get('dph_total', 0):6.2f}/hr | "
              f"score: {score:5.1f} | "
              f"duration: {offer.get('duration', 0)/86400:.0f}d")


def parse_args():
    """인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Launch training on Vast.ai (v3 - console-identical)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 콘솔과 동일하게 검색 (image_tag 포함)
  python vastai_launcher.py --search --image-tag "vastai/pytorch:@vastai-automatic-tag"

  # image_tag 없이 검색 (더 많은 결과)
  python vastai_launcher.py --search --no-image-tag

  # 비교
  python vastai_launcher.py --compare --image-tag "vastai/pytorch:@vastai-automatic-tag"
"""
    )

    parser.add_argument("--docker-image", "-i",
        default="vastai/pytorch:@vastai-automatic-tag")
    
    # 이미지 태그 (핵심!)
    parser.add_argument("--image-tag",
        default=None,
        help="Image tag for filtering (e.g., 'vastai/pytorch:@vastai-automatic-tag')")
    parser.add_argument("--no-image-tag", action="store_true",
        help="Disable image_tag filter (more results)")
    
    # 환경변수
    parser.add_argument("--wandb-key", default=os.getenv("WANDB_API_KEY", ""))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--aws-key", default=os.getenv("AWS_ACCESS_KEY_ID", ""))
    parser.add_argument("--aws-secret", default=os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    parser.add_argument("--aws-region", default=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"))
    parser.add_argument("--s3-path", default=os.getenv("S3_DATA_PATH", ""))
    parser.add_argument("--api-key", default=os.getenv("VAST_API_KEY", ""))

    # GitHub 설정 (프로젝트 clone용)
    parser.add_argument("--gh-token", default=os.getenv("GH_TOKEN", ""))
    parser.add_argument("--gh-user", default=os.getenv("GH_USER", ""))
    parser.add_argument("--gh-repo", default=os.getenv("GH_REPO", ""))
    parser.add_argument("--gh-branch", default=os.getenv("GH_BRANCH", "main"))

    # 동작 모드
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--compare", action="store_true",
        help="Compare with/without image_tag filter")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.api_key:
        print("Error: VAST_API_KEY required")
        sys.exit(1)

    # image_tag 처리
    image_tag = None
    if args.image_tag and not args.no_image_tag:
        image_tag = args.image_tag

    cfg = Config(
        api_key=args.api_key,
        docker_image=args.docker_image,
        image_tag=image_tag,
        gh_token=args.gh_token,
        gh_user=args.gh_user,
        gh_repo=args.gh_repo,
        gh_branch=args.gh_branch,
        env_vars={
            "wandb_key": args.wandb_key,
            "hf_token": args.hf_token,
            "aws_key": args.aws_key,
            "aws_secret": args.aws_secret,
            "aws_region": args.aws_region,
            "s3_path": args.s3_path,
        },
        debug=args.debug,
    )

    # 필수 설정 확인
    if not cfg.gh_token:
        print("Error: GH_TOKEN is required for provisioning. Please set GH_TOKEN env var or --gh-token argument.")
        sys.exit(1)

    # 디버그 출력 (토큰 길이만 표시)
    print(f"[DEBUG] GH_TOKEN length: {len(cfg.gh_token)}")
    print(f"[DEBUG] GH_USER: {cfg.gh_user}")
    print(f"[DEBUG] GH_REPO: {cfg.gh_repo}")
    print(f"[DEBUG] GH_BRANCH: {cfg.gh_branch}")
    if not cfg.gh_user or not cfg.gh_repo:
        print("Error: GH_USER and GH_REPO are required for provisioning.")
        sys.exit(1)

    if args.compare:
        compare_with_without_image_tag(cfg)
    elif args.list:
        list_my_instances(cfg)
    elif args.search:
        search_only(cfg)
    else:
        result = launch(cfg, dry_run=args.dry_run)
        if result is None and not args.dry_run:
            sys.exit(1)


if __name__ == "__main__":
    main()