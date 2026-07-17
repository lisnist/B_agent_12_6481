from typing import Literal

from pydantic import BaseModel, Field


class UploadedFileRef(BaseModel):
    name: str
    path: str
    size: int


class UploadFilePayload(BaseModel):
    name: str = Field(..., min_length=1)
    content_base64: str = Field(..., min_length=1)
    size: int | None = None
    mime_type: str | None = None


class UploadRequest(BaseModel):
    conversation_id: str | None = None
    files: list[UploadFilePayload] = Field(default_factory=list)


class UploadResponse(BaseModel):
    files: list[UploadedFileRef]


class RunRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    conversation_id: str | None = None
    system_prompt: str | None = None
    uploaded_files: list[UploadedFileRef] = Field(default_factory=list)
    uploaded_file_payloads: list[UploadFilePayload] = Field(default_factory=list)
    selected_memory_ids: list[str] = Field(default_factory=list)
    use_global_memory: bool = False
    toolset: str = "basic_tools"
    save_memory: Literal["none", "conversation", "global"] = "conversation"
    llm_mode: Literal["mock", "prompt_json"] | None = None


class RunResponse(BaseModel):
    conversation_id: str
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    status: str
    final_answer: str
    elapsed_ms: float
    output_dir: str
    trace: dict


class ConversationSummary(BaseModel):
    id: str
    title: str
    is_trivial: bool = False
    created_at: str
    updated_at: str
    last_message_at: str | None = None


class ConversationMessage(BaseModel):
    id: str
    role: str
    content: str
    message_order: int
    created_at: str
    status: Literal["pending", "error", "cancelled"] | None = None
    resumable: bool = False
    tool_steps: list[dict] = Field(default_factory=list)
    attachments: list[UploadedFileRef] = Field(default_factory=list)


class ConversationDetail(BaseModel):
    conversation_id: str
    messages: list[ConversationMessage]


class DeleteConversationResponse(BaseModel):
    conversation_id: str
    deleted: bool
    upload_dir_deleted: bool
    output_dir_deleted: bool


class ConversationPromptResponse(BaseModel):
    conversation_id: str
    prompt_id: str
    content: str
    default_content: str
    locked_default: bool = False


class UpdateConversationPromptRequest(BaseModel):
    content: str = Field(..., min_length=1)


class B2SkillRunRequest(BaseModel):
    skill_name: str = Field(..., min_length=1)
    input: dict = Field(default_factory=dict)
    toolset: str = "basic_tools"


class B3ToolCallsPreviewRequest(BaseModel):
    ai_message: dict | None = None
    tool_calls: list = Field(default_factory=list)
    toolset: str = "basic_tools"


class B4ProtocolTestRequest(BaseModel):
    case_id: str = Field(..., min_length=1)


class B5RecallPreviewRequest(BaseModel):
    current_user_input: str = Field(..., min_length=1)
