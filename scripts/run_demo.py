"""
学术论文智能搜索 Agent - 交互式 Demo
"""
from scholar_agent.config import settings
from scholar_agent.utils.logger import setup_logger
from scholar_agent.agent import SearchAgent


def main() -> None:
    setup_logger(settings.output_dir)
    agent = SearchAgent()

    print("=" * 50)
    print("学术论文智能搜索 Agent")
    print("=" * 50)
    print("输入查询（中英文均可），输入 q 退出\n")

    while True:
        try:
            query = input("请输入查询: ").strip()
            if query.lower() == 'q':
                break
            if not query:
                continue
            result = agent.run(query)
            print(f"\n搜索完成，结果已保存至 output/ 目录\n")
        except KeyboardInterrupt:
            print("\n\n退出。")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")


if __name__ == '__main__':
    main()
