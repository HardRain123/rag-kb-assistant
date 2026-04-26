import re


PROCESS_CHANNEL_PATTERNS = [
    re.compile(r"(线上|线下).*(申请|提交|区别|方式)"),
    re.compile(r"(申请方式).*(区别|不同|怎么选)"),
]

PROCESS_ENTRY_PATTERNS = [
    re.compile(r"(怎么|如何).*(申请|理赔|报案)"),
    re.compile(r"(报案|通知).*(下一步|之后|后面)"),
]

PROCESS_GENERAL_PATTERNS = [
    re.compile(r"(申请|理赔).*(流程|步骤)"),
    re.compile(r"(提交).*(申请|理赔)"),
]

SUPPLEMENT_PATTERNS = [
    re.compile(r"(补件|补材料|补资料)"),
    re.compile(r"(材料|资料).*(不全|不够|缺|少|没了|丢了)"),
    re.compile(r"(不全|不够|缺|少).*(材料|资料)"),
    re.compile(r"(材料不全|材料不够|资料不全|资料不够)"),
    re.compile(r"(发票).*(费用清单).*(没|丢|缺)"),
    re.compile(r"(费用清单).*(没|丢|缺)"),
]

BOUNDARY_PATTERNS = [
    re.compile(r"(等待期|免责|医院范围|非约定医院|急诊例外)"),
    re.compile(r"(能赔吗|可以赔吗).*(等待期|医院|急诊|免责)"),
]

CONDITION_PATTERNS = [
    re.compile(r"(谁|哪些人|什么人).*(可以|能).*(申请|理赔)"),
    re.compile(r"(申请|理赔).*(条件|资格)"),
    re.compile(r"(谁来报|谁来申请|谁能报)"),
]

MATERIAL_PATTERNS = [
    re.compile(r"(提交|准备|需要|要).*(材料|资料|发票|费用清单|诊断证明|出院小结)"),
    re.compile(r"(住院|理赔).*(交啥|交什么|交哪些|要交)"),
    re.compile(r"(哪些材料|材料清单|要哪些资料)"),
]

PROCESS_HEADING_KEYWORDS = (
    "流程",
    "步骤",
    "报案",
    "通知",
    "申请方式",
    "线上申请",
    "线下申请",
    "提交申请",
    "线上提交",
    "线下提交",
    "进入审核",
    "支付与到账",
    "SOP",
)

BOUNDARY_KEYWORDS = (
    "等待期",
    "免责",
    "医院范围",
    "非约定医院",
    "急诊例外",
)

CONDITION_KEYWORDS = (
    "申请条件",
    "资格",
    "谁可以申请",
    "谁来申请",
    "谁能报",
)

SUPPLEMENT_KEYWORDS = (
    "补件",
    "材料不全",
    "审核暂停",
    "缺失",
    "补打印",
    "重新提交",
)

MATERIAL_KEYWORDS = (
    "材料清单",
    "基础材料",
    "病历材料",
    "费用材料",
    "发票",
    "费用明细",
    "诊断证明",
    "出院小结",
)

INTENT_PRIORITY_FILTERS = {
    "process": [
        {"chunk_type": "process"},
        {"doc_type": "sop"},
        {"doc_type": "scene"},
    ],
    "process_entry": [
        {"chunk_type": "process"},
        {"doc_type": "sop"},
        {"doc_type": "scene"},
    ],
    "process_channel": [
        {"chunk_type": "process"},
        {"doc_type": "sop"},
        {"doc_type": "scene"},
    ],
    "material": [
        {"chunk_type": "material"},
        {"doc_type": "rule"},
        {"doc_type": "faq"},
        {"doc_type": "sop"},
    ],
    "supplement": [
        {"chunk_type": "supplement"},
        {"doc_type": "scene"},
        {"doc_type": "rule"},
        {"doc_type": "sop"},
    ],
    "condition": [
        {"chunk_type": "condition"},
        {"doc_type": "rule"},
        {"doc_type": "faq"},
    ],
    "boundary": [
        {"chunk_type": "boundary"},
        {"doc_type": "rule"},
        {"doc_type": "faq"},
        {"doc_type": "scene"},
    ],
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def classify_query_intent(question: str) -> str | None:
    normalized = _normalize_text(question)
    if not normalized:
        return None

    for pattern in PROCESS_CHANNEL_PATTERNS:
        if pattern.search(normalized):
            return "process_channel"

    for pattern in PROCESS_ENTRY_PATTERNS:
        if pattern.search(normalized):
            return "process_entry"

    for pattern in PROCESS_GENERAL_PATTERNS:
        if pattern.search(normalized):
            return "process"

    for pattern in SUPPLEMENT_PATTERNS:
        if pattern.search(normalized):
            return "supplement"

    for pattern in BOUNDARY_PATTERNS:
        if pattern.search(normalized):
            return "boundary"

    for pattern in CONDITION_PATTERNS:
        if pattern.search(normalized):
            return "condition"

    for pattern in MATERIAL_PATTERNS:
        if pattern.search(normalized):
            return "material"

    return None


def infer_doc_type(file_name: str, doc_type_override: str | None = None) -> str:
    if doc_type_override:
        return doc_type_override

    normalized = file_name.lower()
    if "faq" in normalized:
        return "faq"
    if "sop" in normalized and "典型工单场景" in file_name:
        return "scene"
    if "sop" in normalized:
        return "sop"
    if "规则说明" in file_name:
        return "rule"
    return "default"


def extract_heading_path(chunk_text: str) -> str:
    headings: list[str] = []
    for line in chunk_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue

        heading = stripped.lstrip("#").strip()
        if heading:
            headings.append(heading)

    return " > ".join(headings)


def infer_chunk_type(heading_path: str, chunk_text: str) -> str:
    heading_segments = [
        segment.strip()
        for segment in (heading_path or "").split(" > ")
        if segment.strip()
    ]
    body_text = chunk_text or ""

    # Prefer the most specific heading segment first so document-level titles
    # like “材料清单与医院范围” do not override a more precise child heading.
    texts = list(reversed(heading_segments)) + [body_text]

    for text in texts:
        if _contains_any_keyword(text, SUPPLEMENT_KEYWORDS):
            return "supplement"
        if _contains_any_keyword(text, CONDITION_KEYWORDS):
            return "condition"
        if _contains_any_keyword(text, BOUNDARY_KEYWORDS):
            return "boundary"
        if _contains_any_keyword(text, MATERIAL_KEYWORDS):
            return "material"
        if _contains_any_keyword(text, PROCESS_HEADING_KEYWORDS):
            return "process"

    return "unknown"


def get_priority_filters_for_intent(intent: str | None) -> list[dict[str, str]]:
    return INTENT_PRIORITY_FILTERS.get(intent or "", [])


def is_process_intent(intent: str | None) -> bool:
    return intent in {"process", "process_entry", "process_channel"}


def is_process_doc(
    file_name: str | None = None, doc_type: str | None = None
) -> bool:
    if doc_type in {"sop", "scene"}:
        return True
    if not file_name:
        return False

    return infer_doc_type(file_name) in {"sop", "scene"}


def is_process_chunk(
    snippet: str, heading_path: str | None = None, chunk_type: str | None = None
) -> bool:
    if chunk_type == "process":
        return True

    candidate = heading_path or ""
    if _contains_any_keyword(candidate, PROCESS_HEADING_KEYWORDS):
        return True

    for line in snippet.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if not stripped:
            continue

        if _contains_any_keyword(stripped, PROCESS_HEADING_KEYWORDS):
            return True

    return False
