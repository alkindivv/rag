# 🚨 ANTI-OVERENGINEERING RULES
**Comprehensive Guidelines to Prevent Complexity Creep and Maintain KISS Principle**

## 🎯 CORE MISSION
**NEVER make simple things complicated without significant impact.**

These rules MUST be followed to prevent AI assistants from creating overengineered solutions that violate the KISS principle and make maintenance nightmares.

---

## 🔥 CRITICAL ANTI-PATTERNS - NEVER DO THESE

### ❌ **FILE PROLIFERATION SYNDROME**
**RULE**: If functionality can be achieved in 1 file, NEVER create multiple files.

**VIOLATIONS TO AVOID:**
- Creating 6 files when 1 file works
- "Separation of concerns" taken to absurd extremes
- Building "modular architecture" for simple tasks
- Creating files "for future extensibility"

src/services/simple_legal_chunker.py  # 346 lines in 1 file
```

### ❌ **ABSTRACTION OVERLOAD**
**RULE**: Don't create abstractions unless they solve REAL duplication.

**VIOLATIONS TO AVOID:**
- Creating base classes for 1-2 implementations
- Building "extensible frameworks" for specific use cases
- Adding abstraction layers "for flexibility"
- Creating interfaces with single implementations

**EXAMPLE OF VIOLATION:**
```python
# DON'T DO THIS:
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self): pass

class AbstractLegalChunker(BaseChunker):
    @abstractmethod
    def detect_structure(self): pass

class LegalChunkerInterface(Protocol):
    def chunk_document(self): pass

# DO THIS INSTEAD:
class Chunker:
    def chunk_document(self, text, metadata):
        # Direct implementation
```

### ❌ **CONFIGURATION EXPLOSION**
**RULE**: Don't create configuration for things that don't need to vary.

**VIOLATIONS TO AVOID:**
- Creating config classes for static values
- Building "flexible configuration systems" for fixed behavior
- Adding environment variables for constants
- Creating config hierarchies

**EXAMPLE OF VIOLATION:**
```python
# DON'T DO THIS:
class ChunkingConfig:
    DEFAULT_MAX_TOKENS = int(os.getenv("CHUNKING_MAX_TOKENS", "1500"))
    AYAT_SPLIT_THRESHOLD = int(os.getenv("CHUNKING_AYAT_SPLIT", "1200"))
    # 50+ more configurable options...

# DO THIS INSTEAD:
max_tokens = 1500  # Simple constant
```

### ❌ **DATACLASS/MODEL OVERENGINEERING**
**RULE**: Use simple data structures unless complex behavior is needed.

**VIOLATIONS TO AVOID:**
- Creating complex dataclasses with methods
- Building object hierarchies for data
- Adding validation logic to data models
- Creating "rich domain objects" for simple data

**EXAMPLE OF VIOLATION:**
```python
# DON'T DO THIS:
@dataclass
class LegalChunk:
    chunk_id: str
    content: str
    chunk_type: ChunkType
    hierarchical_context: str
    citation_path: str
    position: Position
    document_metadata: Dict[str, Any]
    semantic_keywords: List[str]
    # 20+ fields and methods...

# DO THIS INSTEAD:
@dataclass
class SimpleChunk:
    chunk_id: str
    content: str
    citation: str
    keywords: List[str]
    tokens: int = 0
```

---

## ✅ KISS ENFORCEMENT RULES

### **RULE 1: THE ONE-FILE TEST**
**Before creating multiple files, ask:**
- Can this be done in one file under 500 lines?
- Is the complexity ACTUALLY necessary?
- Would splitting it make it HARDER to understand?

**If ANY answer is YES → Use one file**

### **RULE 2: THE 15-MINUTE RULE**
**Any solution should be understandable by a developer in 15 minutes or less.**

**If it takes longer:**
- The solution is too complex
- Break it down further
- Remove unnecessary abstractions
- Simplify the approach

### **RULE 3: THE DUPLICATION THRESHOLD**
**Don't abstract until you have 3+ real instances of duplication.**

**Guidelines:**
- 1 instance = No abstraction needed
- 2 instances = Copy is fine, watch for patterns
- 3+ instances = Consider abstraction
- Never abstract speculatively

### **RULE 4: THE NET BENEFIT RULE**
**Every abstraction/pattern MUST provide measurable benefit.**

**Measurable benefits:**
- ✅ Reduces actual code duplication (lines saved)
- ✅ Simplifies maintenance (fewer places to change)
- ✅ Improves performance (measurable gains)

**NOT benefits:**
- ❌ "Makes it more extensible"
- ❌ "Follows best practices"
- ❌ "Better architecture"
- ❌ "More professional"

### **RULE 5: THE DELETION TEST**
**Regularly ask: What can we DELETE?**

**Before adding new code:**
- Can existing code be simplified?
- Are there unused abstractions?
- Can multiple classes be merged?
- Are there redundant patterns?

---

## 🚫 FORBIDDEN PHRASES & JUSTIFICATIONS

### **BANNED JUSTIFICATIONS:**
- ❌ "For future extensibility"
- ❌ "Following best practices"
- ❌ "Separation of concerns"
- ❌ "Clean architecture"
- ❌ "Enterprise patterns"
- ❌ "Scalable design"
- ❌ "Maintainable code" (when it's actually harder to maintain)

### **REQUIRED JUSTIFICATIONS:**
- ✅ "Removes X lines of duplicate code"
- ✅ "Solves actual performance problem"
- ✅ "Eliminates real maintenance burden"
- ✅ "Required by external constraints"

---

## 📊 COMPLEXITY METRICS & LIMITS

### **FILE LIMITS:**
- ✅ Under 500 lines = Good
- ⚠️ 500-800 lines = Acceptable if justified
- ❌ Over 800 lines = Must be split (unless single algorithm)

### **CLASS LIMITS:**
- ✅ Under 200 lines = Good
- ⚠️ 200-400 lines = Acceptable if single responsibility
- ❌ Over 400 lines = Too complex

### **METHOD LIMITS:**
- ✅ Under 50 lines = Good
- ⚠️ 50-100 lines = Acceptable if single task
- ❌ Over 100 lines = Break down

### **DEPENDENCY LIMITS:**
- ✅ 0-3 dependencies = Good
- ⚠️ 4-7 dependencies = Monitor closely
- ❌ 8+ dependencies = Too coupled

---

## 🎯 DECISION FRAMEWORK

### **BEFORE WRITING CODE, ASK:**

1. **Can this be a simple function instead of a class?**
2. **Can this be done in existing files instead of new ones?**
3. **What's the simplest thing that could possibly work?**
4. **Am I solving a real problem or creating architecture?**
5. **Would a junior developer understand this in 15 minutes?**

### **RED FLAGS - STOP AND RECONSIDER:**
- Creating more than 2 new files
- Adding more than 500 new lines of code
- Creating abstract base classes
- Building "flexible" systems
- Adding configuration for everything
- Creating models with many fields
- Using design patterns "for the pattern"

### **GREEN FLAGS - PROCEED:**
- Replacing existing complex code with simpler code
- Eliminating code duplication
- Fixing actual performance problems
- Meeting specific external requirements
- Making debugging easier

---

## 🛠️ PRACTICAL IMPLEMENTATION GUIDELINES

### **FOR CHUNKING/PROCESSING TASKS:**
```python
# GOOD: Simple, direct implementation
class SimpleProcessor:
    def process(self, input_data):
        # Do the work directly
        result = self._parse(input_data)
        return self._format(result)

# BAD: Over-abstracted
class AbstractProcessor(ABC):
    @abstractmethod
    def validate_input(self): pass
    @abstractmethod
    def process_data(self): pass
    @abstractmethod
    def format_output(self): pass
```

### **FOR DATA HANDLING:**
```python
# GOOD: Simple data structure
@dataclass
class Result:
    content: str
    metadata: Dict[str, Any]

# BAD: Over-modeled domain
class RichResult:
    def __init__(self):
        self._content = None
        self._metadata = {}

    @property
    def content(self): return self._content

    def validate(self): pass
    def enrich(self): pass
    def transform(self): pass
```

### **FOR CONFIGURATION:**
```python
# GOOD: Simple constants or basic config
MAX_TOKENS = 1500
PATTERNS = {
    'pasal': r'Pasal\s+(\d+)',
    'bab': r'BAB\s+([IVX]+)'
}

# BAD: Over-configured system
class ConfigManager:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.validators = []
        self.transformers = []

    def load_config(self, env, profile): pass
    def validate_config(self): pass
```

---

## 🔍 CODE REVIEW CHECKLIST

### **BEFORE MERGING CODE, VERIFY:**

**SIMPLICITY CHECKS:**
- [ ] Could this be simpler and still work?
- [ ] Are all classes/files actually necessary?
- [ ] Can any abstractions be removed?
- [ ] Is the solution easy to understand?

**KISS COMPLIANCE:**
- [ ] Does it follow the one-file test?
- [ ] Does it pass the 15-minute rule?
- [ ] Are all dependencies justified?
- [ ] Would deletion make it simpler?

**BENEFIT VERIFICATION:**
- [ ] What specific problem does this solve?
- [ ] How much code duplication does it eliminate?
- [ ] What maintenance burden does it reduce?
- [ ] Are the benefits measurable?

**ANTI-PATTERN DETECTION:**
- [ ] No file proliferation syndrome?
- [ ] No abstraction overload?
- [ ] No configuration explosion?
- [ ] No dataclass overengineering?

---

## 💡 REMEMBER THE GOAL

**The goal is NOT to write "enterprise-grade" or "scalable" code.**

**The goal IS to write code that:**
- ✅ Solves the actual problem
- ✅ Is easy to understand
- ✅ Is easy to modify
- ✅ Works reliably
- ✅ Can be maintained by anyone

**SIMPLE WORKING CODE > COMPLEX PERFECT CODE**

---

## 🚨 FINAL WARNING

**If you find yourself:**
- Creating "extensible architectures"
- Building "flexible frameworks"
- Adding "separation of concerns"
- Creating "clean abstractions"
- Following "enterprise patterns"

**STOP. You're overengineering.**

**Instead, ask:**
- What's the simplest thing that works?
- Can I do this in fewer files?
- Can I do this with less code?
- Will this be easier to debug?

**Remember: Code is a liability, not an asset. Less code is better code.**

---

*"Simplicity is the ultimate sophistication." - Leonardo da Vinci*

*"Any fool can make things bigger and more complex. It takes a touch of genius - and a lot of courage - to move in the opposite direction." - E.F. Schumacher*
