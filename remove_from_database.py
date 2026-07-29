"""
从底库中删除指定人名的记录

用法:
  # 按人名删除
  python remove_from_database.py --name xiaoming

  # 删除多个
  python remove_from_database.py --name zhangsan --name lisi

  # 指定底库路径
  python remove_from_database.py --name xiaoming --database my_db.npz

  # 预览要删的人（不实际删除，只显示信息）
  python remove_from_database.py --name xiaoming --dry-run
"""

import os
import argparse
import numpy as np

# ========== 配置 ==========
DEFAULT_DB = r"C:\yolo_learning\face_database.npz"
# ==========================


def load_database(db_path: str) -> dict:
    """加载底库，返回 {name: embedding}。"""
    if not os.path.exists(db_path):
        print(f"[ERROR] 底库不存在: {db_path}")
        return None

    data = np.load(db_path, allow_pickle=True)
    names = data['names']
    embeddings = data['embeddings']
    database = {str(name): emb for name, emb in zip(names, embeddings)}
    print(f"[INFO] 已加载底库: {len(database)} 人")
    return database


def save_database(database: dict, db_path: str):
    """保存底库。"""
    names = np.array(list(database.keys()))
    embeddings = np.array(list(database.values()))
    np.savez(db_path, names=names, embeddings=embeddings)
    print(f"[OK] 底库已保存: {db_path}  ({len(database)} 人)")


def main():
    parser = argparse.ArgumentParser(
        description="从人脸底库中删除记录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python remove_from_database.py --name xiaoming
  python remove_from_database.py --name zhangsan --name lisi
  python remove_from_database.py --name xiaoming --dry-run
        """,
    )

    parser.add_argument('--name', type=str, required=True, action='append',
                        help="要删除的人名（可重复使用，如 --name a --name b）")
    parser.add_argument('--database', type=str, default=DEFAULT_DB,
                        help=f"底库路径（默认: {DEFAULT_DB}）")
    parser.add_argument('--dry-run', action='store_true',
                        help="预览模式：只显示匹配结果，不实际删除")

    args = parser.parse_args()

    # ---- 加载底库 ----
    database = load_database(args.database)
    if database is None:
        return

    # ---- 查找并删除 ----
    before_count = len(database)
    removed = []
    not_found = []

    for name in args.name:
        if name in database:
            del database[name]
            removed.append(name)
        else:
            not_found.append(name)

    # ---- 输出结果 ----
    print()

    if not_found:
        print(f"[WARN] 底库中不存在: {', '.join(not_found)}")
        # 模糊匹配提示
        for name in not_found:
            matches = [n for n in database if name.lower() in n.lower()]
            if matches:
                print(f"       你可能想找: {', '.join(matches[:5])}")

    if removed:
        for name in removed:
            print(f"[RM] 已删除: {name}")

    print()

    if args.dry_run:
        print(f"[DRY-RUN] 预览模式，未实际修改底库")
        print(f"  将会删除: {len(removed)} 人")
        print(f"  剩余:      {before_count - len(removed)} 人")
    elif removed:
        save_database(database, args.database)

    # ---- 汇总 ----
    print(f"\n{'='*50}")
    print(f"删除完成!")
    print(f"  原有:    {before_count} 人")
    print(f"  删除:    {len(removed)} 人")
    print(f"  未找到:  {len(not_found)} 人")
    print(f"  剩余:    {len(database)} 人")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
