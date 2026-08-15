from pipeline.token_matching.eval.run_eval import run_eval


def main():
    result = run_eval()
    print(result["status"])


if __name__ == "__main__":
    main()
