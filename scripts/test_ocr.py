import sys

import torch
from paddleocr import PaddleOCR


def main() -> None:
    image_path = sys.argv[1]

    print("PyTorch CUDA:", torch.cuda.is_available())

    ocr = PaddleOCR(
        lang="en",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    results = ocr.predict(image_path)

    for result in results:
        result.print()
        print(result.json)


if __name__ == "__main__":
    main()
