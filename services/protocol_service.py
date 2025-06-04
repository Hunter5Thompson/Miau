# services/protocol_service.py
"""
Protocol Formatting Service

Konvertiert Transkriptionen in strukturierte Gemeinderatsprotokolle.
Wie ein erfahrener Protokollführer, der aus chaotischen Gesprächsnotizen
ein ordentliches, offizielles Dokument macht.
"""

import re
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path

from services.transcription_service import TranscriptionResult
from services.alignment_service import AlignmentResult
from services.diarization_service import DiarizationResult, SpeakerSegment

logger = logging.getLogger(__name__)


class ProtocolFormat(Enum):
    """Available protocol formats"""
    MUNICIPAL_COUNCIL = "municipal_council"     # Standard Gemeinderatsprotokoll
    MEETING_MINUTES = "meeting_minutes"         # General meeting minutes
    INTERVIEW = "interview"                     # Interview format
    CONFERENCE = "conference"                   # Conference proceedings
    CUSTOM = "custom"                           # Custom format


class TimestampFormat(Enum):
    """Timestamp display formats"""
    NONE = "none"                              # No timestamps
    SIMPLE = "simple"                          # (MM:SS)
    DETAILED = "detailed"                      # (HH:MM:SS)
    RELATIVE = "relative"                      # (+MM:SS from start)


@dataclass
class ProtocolConfig:
    """Configuration for protocol formatting"""
    
    # Format settings
    format_type: ProtocolFormat = ProtocolFormat.MUNICIPAL_COUNCIL
    timestamp_format: TimestampFormat = TimestampFormat.DETAILED
    
    # Content organization
    group_by_speaker: bool = False
    group_by_topic: bool = True
    min_topic_duration: float = 30.0  # Minimum seconds for a topic
    
    # Text processing
    capitalize_sentences: bool = True
    add_punctuation: bool = True
    remove_filler_words: bool = True
    
    # Speaker handling
    use_speaker_names: bool = True
    merge_consecutive_speaker_segments: bool = True
    min_speaker_segment: float = 2.0  # Minimum seconds for speaker change
    
    # Header/Footer
    include_header: bool = True
    include_footer: bool = True
    include_statistics: bool = True
    
    # Advanced features
    detect_decisions: bool = True      # Detect voting/decisions
    detect_action_items: bool = True   # Detect assignments/tasks
    highlight_important: bool = True   # Highlight key phrases
    
    @classmethod
    def get_municipal_config(cls) -> 'ProtocolConfig':
        """Configuration for municipal council protocols"""
        return cls(
            format_type=ProtocolFormat.MUNICIPAL_COUNCIL,
            timestamp_format=TimestampFormat.DETAILED,
            group_by_topic=True,
            capitalize_sentences=True,
            add_punctuation=True,
            remove_filler_words=True,
            use_speaker_names=True,
            include_header=True,
            include_footer=True,
            detect_decisions=True,
            detect_action_items=True
        )
    
    @classmethod
    def get_meeting_config(cls) -> 'ProtocolConfig':
        """Configuration for general meeting minutes"""
        return cls(
            format_type=ProtocolFormat.MEETING_MINUTES,
            timestamp_format=TimestampFormat.SIMPLE,
            group_by_speaker=True,
            capitalize_sentences=True,
            add_punctuation=True,
            remove_filler_words=False,  # Keep natural speech
            use_speaker_names=True,
            include_header=True,
            include_footer=False,
            detect_action_items=True
        )
    
    @classmethod
    def get_interview_config(cls) -> 'ProtocolConfig':
        """Configuration for interview transcripts"""
        return cls(
            format_type=ProtocolFormat.INTERVIEW,
            timestamp_format=TimestampFormat.NONE,
            group_by_speaker=True,
            capitalize_sentences=True,
            add_punctuation=True,
            remove_filler_words=False,  # Keep natural speech
            use_speaker_names=True,
            merge_consecutive_speaker_segments=False,  # Keep all speaker changes
            include_header=False,
            include_footer=False
        )


@dataclass
class TopicSegment:
    """A detected topic segment in the protocol"""
    title: str
    start_time: float
    end_time: float
    speaker_segments: List[SpeakerSegment]
    confidence: float = 0.0
    
    def duration(self) -> float:
        """Duration of the topic in seconds"""
        return self.end_time - self.start_time
    
    def word_count(self) -> int:
        """Total words in this topic"""
        return sum(seg.word_count() for seg in self.speaker_segments)


@dataclass
class DecisionItem:
    """A detected decision or vote in the protocol"""
    text: str
    timestamp: float
    decision_type: str  # "vote", "decision", "resolution"
    participants: List[str]
    confidence: float = 0.0


@dataclass
class ActionItem:
    """A detected action item or assignment"""
    text: str
    timestamp: float
    assignee: Optional[str]
    deadline: Optional[str]
    confidence: float = 0.0


@dataclass
class ProtocolResult:
    """Complete formatted protocol result"""
    formatted_text: str
    metadata: Dict[str, Any]
    topics: List[TopicSegment]
    decisions: List[DecisionItem]
    action_items: List[ActionItem]
    
    # Processing statistics
    processing_time: float
    original_duration: float
    total_words: int
    total_speakers: int
    
    def save_to_file(self, file_path: Path) -> None:
        """Save protocol to file"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.formatted_text)
        
        logger.info(f"📄 Protocol saved: {file_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get protocol statistics"""
        return {
            "total_duration_minutes": self.original_duration / 60,
            "total_words": self.total_words,
            "total_speakers": self.total_speakers,
            "topics_count": len(self.topics),
            "decisions_count": len(self.decisions),
            "action_items_count": len(self.action_items),
            "processing_time_seconds": self.processing_time,
            "words_per_minute": (self.total_words / (self.original_duration / 60)) if self.original_duration > 0 else 0
        }


class TextProcessor:
    """Handles text processing and cleanup"""
    
    # Common German filler words for municipal meetings
    GERMAN_FILLER_WORDS = {
        "äh", "ähm", "eh", "ehm", "mhm", "hmm", "ja", "naja", "also", 
        "halt", "eben", "eigentlich", "irgendwie", "sozusagen", "gewissermaßen"
    }
    
    @staticmethod
    def clean_text(text: str, config: ProtocolConfig) -> str:
        """Clean and process text according to configuration"""
        if not text:
            return ""
        
        processed = text.strip()
        
        # Remove filler words
        if config.remove_filler_words:
            processed = TextProcessor._remove_filler_words(processed)
        
        # Add punctuation
        if config.add_punctuation:
            processed = TextProcessor._add_punctuation(processed)
        
        # Capitalize sentences
        if config.capitalize_sentences:
            processed = TextProcessor._capitalize_sentences(processed)
        
        return processed
    
    @staticmethod
    def _remove_filler_words(text: str) -> str:
        """Remove common German filler words"""
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Remove punctuation for comparison
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word not in TextProcessor.GERMAN_FILLER_WORDS:
                cleaned_words.append(word)
        
        return " ".join(cleaned_words)
    
    @staticmethod
    def _add_punctuation(text: str) -> str:
        """Add basic punctuation to text"""
        if not text:
            return text
        
        # Add period at end if missing
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Add commas before common conjunctions
        text = re.sub(r'\s+(und|oder|aber|doch|jedoch|dennoch)\s+', r', \1 ', text)
        
        return text
    
    @staticmethod
    def _capitalize_sentences(text: str) -> str:
        """Capitalize sentence beginnings"""
        if not text:
            return text
        
        # Split by sentence endings and capitalize
        sentences = re.split(r'([.!?]+)', text)
        
        for i in range(0, len(sentences), 2):  # Every other element is text
            if sentences[i].strip():
                sentences[i] = sentences[i].strip()
                if sentences[i]:
                    sentences[i] = sentences[i][0].upper() + sentences[i][1:]
        
        return ''.join(sentences)


class TopicDetector:
    """Detects topics and agenda items in transcriptions"""
    
    # German patterns for topic detection
    TOPIC_PATTERNS = [
        r"(?i)TOP\s*(\d+)\s*[:.-]?\s*(.*)",                    # TOP 1: Description
        r"(?i)Tagesordnungspunkt\s*(\d+)\s*[:.-]?\s*(.*)",     # Tagesordnungspunkt 1: Description
        r"(?i)Punkt\s*(\d+)\s*[:.-]?\s*(.*)",                  # Punkt 1: Description
        r"(?i)Agenda\s*(?:punkt)?\s*(\d+)\s*[:.-]?\s*(.*)",    # Agenda punkt 1: Description
        r"(?i)Thema\s*(\d+)?\s*[:.-]?\s*(.*)",                 # Thema: Description
        r"(?i)Wir\s+kommen\s+(?:jetzt\s+)?zu\s+(.*)",          # Wir kommen jetzt zu...
        r"(?i)Als\s+nächstes\s+(?:behandeln\s+wir\s+)?(.*)",   # Als nächstes behandeln wir...
    ]
    
    @staticmethod
    def detect_topics(segments: List[SpeakerSegment], min_duration: float = 30.0) -> List[TopicSegment]:
        """Detect topics in speaker segments"""
        topics = []
        current_topic = None
        current_segments = []
        
        for segment in sorted(segments, key=lambda x: x.start):
            text = segment.text.strip()
            
            # Check for topic indicators
            topic_match = TopicDetector._find_topic_match(text)
            
            if topic_match:
                # Save previous topic if it exists and meets minimum duration
                if current_topic and current_segments:
                    topic_duration = current_segments[-1].end - current_segments[0].start
                    if topic_duration >= min_duration:
                        topics.append(TopicSegment(
                            title=current_topic,
                            start_time=current_segments[0].start,
                            end_time=current_segments[-1].end,
                            speaker_segments=current_segments.copy(),
                            confidence=0.8
                        ))
                
                # Start new topic
                current_topic = topic_match
                current_segments = [segment]
            else:
                # Add to current topic
                if current_topic:
                    current_segments.append(segment)
                else:
                    # No topic yet, create default
                    if not topics:
                        current_topic = "Eröffnung und Begrüßung"
                        current_segments = [segment]
        
        # Don't forget the last topic
        if current_topic and current_segments:
            topic_duration = current_segments[-1].end - current_segments[0].start
            if topic_duration >= min_duration:
                topics.append(TopicSegment(
                    title=current_topic,
                    start_time=current_segments[0].start,
                    end_time=current_segments[-1].end,
                    speaker_segments=current_segments,
                    confidence=0.8
                ))
        
        logger.info(f"🔍 Detected {len(topics)} topics")
        return topics
    
    @staticmethod
    def _find_topic_match(text: str) -> Optional[str]:
        """Find topic pattern match in text"""
        for pattern in TopicDetector.TOPIC_PATTERNS:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) >= 2:
                    # Pattern with number and description
                    number = match.group(1)
                    description = match.group(2).strip()
                    if description and len(description.split()) <= 15:  # Reasonable title length
                        return f"TOP {number}. {description}" if number else description
                elif len(match.groups()) >= 1:
                    # Pattern with just description
                    description = match.group(1).strip()
                    if description and len(description.split()) <= 15:
                        return description
        
        return None


class DecisionDetector:
    """Detects decisions and votes in transcriptions"""
    
    DECISION_PATTERNS = [
        r"(?i)(?:ich\s+)?(?:stelle|bringe)\s+(?:den\s+)?(?:antrag|vorschlag)\s+zur\s+abstimmung",
        r"(?i)wer\s+ist\s+dafür\s*[?\.]?\s*wer\s+ist\s+dagegen",
        r"(?i)(?:wird\s+)?(?:einstimmig\s+)?(?:angenommen|beschlossen|abgelehnt)",
        r"(?i)abstimmung\s+(?:über|zu|für)",
        r"(?i)(?:stimmen\s+)?(?:mit\s+)?(\d+)\s+(?:zu\s+)?(\d+)\s+(?:stimmen\s+)?(?:angenommen|abgelehnt)",
        r"(?i)mehrheitlich\s+(?:angenommen|beschlossen|abgelehnt)",
        r"(?i)(?:der\s+)?(?:stadt)?rat\s+beschließt",
    ]
    
    @staticmethod
    def detect_decisions(segments: List[SpeakerSegment]) -> List[DecisionItem]:
        """Detect decisions and votes in segments"""
        decisions = []
        
        for segment in segments:
            text = segment.text.strip()
            
            for pattern in DecisionDetector.DECISION_PATTERNS:
                if re.search(pattern, text):
                    # Determine decision type
                    if "abstimmung" in text.lower() or "stimme" in text.lower():
                        decision_type = "vote"
                    elif "beschluss" in text.lower() or "beschließt" in text.lower():
                        decision_type = "resolution"
                    else:
                        decision_type = "decision"
                    
                    decision = DecisionItem(
                        text=text,
                        timestamp=segment.start,
                        decision_type=decision_type,
                        participants=[segment.speaker_id] if hasattr(segment, 'speaker_id') else [],
                        confidence=0.7
                    )
                    decisions.append(decision)
                    break
        
        logger.info(f"⚖️ Detected {len(decisions)} decisions")
        return decisions


class ActionItemDetector:
    """Detects action items and assignments"""
    
    ACTION_PATTERNS = [
        r"(?i)(?:herr|frau)\s+\w+\s+(?:soll|wird|übernimmt)",
        r"(?i)(?:die\s+)?verwaltung\s+(?:soll|wird|übernimmt)",
        r"(?i)(?:bis\s+(?:zum\s+)?(\d+\.?\d*\.?\d*)|(?:in\s+)?(\d+)\s+(?:wochen?|monaten?|tagen?))",
        r"(?i)(?:wird\s+)?(?:beauftragt|gebeten|ersucht)",
        r"(?i)(?:action\s+item|aufgabe|auftrag|todo)",
        r"(?i)(?:wer\s+)?(?:kümmert\s+sich|übernimmt\s+das|macht\s+das)",
    ]
    
    @staticmethod
    def detect_action_items(segments: List[SpeakerSegment]) -> List[ActionItem]:
        """Detect action items and assignments"""
        actions = []
        
        for segment in segments:
            text = segment.text.strip()
            
            for pattern in ActionItemDetector.ACTION_PATTERNS:
                if re.search(pattern, text):
                    # Try to extract assignee
                    assignee = ActionItemDetector._extract_assignee(text)
                    
                    # Try to extract deadline
                    deadline = ActionItemDetector._extract_deadline(text)
                    
                    action = ActionItem(
                        text=text,
                        timestamp=segment.start,
                        assignee=assignee,
                        deadline=deadline,
                        confidence=0.6
                    )
                    actions.append(action)
                    break
        
        logger.info(f"📋 Detected {len(actions)} action items")
        return actions
    
    @staticmethod
    def _extract_assignee(text: str) -> Optional[str]:
        """Try to extract assignee from text"""
        # Look for names or roles
        name_patterns = [
            r"(?i)(?:herr|frau)\s+(\w+)",
            r"(?i)(verwaltung|bürgermeister|stadtrat)",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def _extract_deadline(text: str) -> Optional[str]:
        """Try to extract deadline from text"""
        deadline_patterns = [
            r"(?i)bis\s+(?:zum\s+)?(\d+\.\d+\.\d+)",
            r"(?i)bis\s+(?:zum\s+)?(\d+\.\d+\.)",
            r"(?i)(?:in\s+)?(\d+)\s+(wochen?|monaten?|tagen?)",
        ]
        
        for pattern in deadline_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return None


class ProtocolFormatter:
    """Main protocol formatter"""
    
    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.text_processor = TextProcessor()
        self.topic_detector = TopicDetector()
        self.decision_detector = DecisionDetector() 
        self.action_detector = ActionItemDetector()
    
    def format_protocol(
        self,
        diarization_result: DiarizationResult,
        municipality: str,
        meeting_date: str,
        meeting_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProtocolResult:
        """Format complete protocol from diarization result"""
        start_time = time.time()
        
        logger.info(f"📄 Formatting protocol: {self.config.format_type.value}")
        
        # Process segments
        processed_segments = self._process_segments(diarization_result.segments)
        
        # Detect structure elements
        topics = self.topic_detector.detect_topics(processed_segments, self.config.min_topic_duration)
        decisions = self.decision_detector.detect_decisions(processed_segments)
        action_items = self.action_detector.detect_action_items(processed_segments)
        
        # Generate formatted text
        formatted_text = self._generate_formatted_text(
            processed_segments, topics, decisions, action_items,
            municipality, meeting_date, meeting_type, metadata
        )
        
        processing_time = time.time() - start_time
        
        # Create result
        result = ProtocolResult(
            formatted_text=formatted_text,
            metadata=metadata or {},
            topics=topics,
            decisions=decisions,
            action_items=action_items,
            processing_time=processing_time,
            original_duration=diarization_result.total_duration,
            total_words=sum(seg.word_count() for seg in processed_segments),
            total_speakers=diarization_result.speaker_count
        )
        
        logger.info(f"✅ Protocol formatted in {processing_time:.2f}s")
        return result
    
    def _process_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Process and clean segments"""
        processed = []
        
        for segment in segments:
            # Clean text
            cleaned_text = self.text_processor.clean_text(segment.text, self.config)
            
            if cleaned_text.strip():  # Only keep non-empty segments
                processed_segment = SpeakerSegment(
                    speaker_id=segment.speaker_id,
                    start=segment.start,
                    end=segment.end,
                    text=cleaned_text,
                    confidence=segment.confidence,
                    words=segment.words
                )
                processed.append(processed_segment)
        
        # Merge consecutive segments from same speaker if configured
        if self.config.merge_consecutive_speaker_segments:
            processed = self._merge_consecutive_segments(processed)
        
        return processed
    
    def _merge_consecutive_segments(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Merge consecutive segments from the same speaker"""
        if not segments:
            return segments
        
        merged = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # Check if should merge
            same_speaker = current_segment.speaker_id == next_segment.speaker_id
            gap = next_segment.start - current_segment.end
            close_enough = gap <= self.config.min_speaker_segment
            
            if same_speaker and close_enough:
                # Merge segments
                current_segment = SpeakerSegment(
                    speaker_id=current_segment.speaker_id,
                    start=current_segment.start,
                    end=next_segment.end,
                    text=f"{current_segment.text} {next_segment.text}".strip(),
                    confidence=current_segment.confidence,
                    words=None  # Merged segments lose word-level info
                )
            else:
                # Save current and start new
                merged.append(current_segment)
                current_segment = next_segment
        
        # Don't forget the last segment
        merged.append(current_segment)
        
        logger.info(f"🔗 Merged segments: {len(segments)} -> {len(merged)}")
        return merged
    
    def _generate_formatted_text(
        self,
        segments: List[SpeakerSegment],
        topics: List[TopicSegment],
        decisions: List[DecisionItem], 
        action_items: List[ActionItem],
        municipality: str,
        meeting_date: str,
        meeting_type: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generate the final formatted protocol text"""
        
        parts = []
        
        # Header
        if self.config.include_header:
            header = self._generate_header(municipality, meeting_date, meeting_type, metadata)
            parts.append(header)
        
        # Main content
        if self.config.group_by_topic and topics:
            content = self._generate_topic_based_content(topics)
        elif self.config.group_by_speaker:
            content = self._generate_speaker_based_content(segments)
        else:
            content = self._generate_chronological_content(segments)
        
        parts.append(content)
        
        # Decisions section
        if decisions and self.config.detect_decisions:
            decisions_section = self._generate_decisions_section(decisions)
            parts.append(decisions_section)
        
        # Action items section
        if action_items and self.config.detect_action_items:
            actions_section = self._generate_actions_section(action_items)
            parts.append(actions_section)
        
        # Footer
        if self.config.include_footer:
            footer = self._generate_footer(municipality, meeting_date, segments, metadata)
            parts.append(footer)
        
        return "\n\n".join(parts)
    
    def _generate_header(
        self, 
        municipality: str, 
        meeting_date: str, 
        meeting_type: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generate protocol header"""
        
        try:
            date_obj = datetime.strptime(meeting_date, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%d.%m.%Y")
        except ValueError:
            formatted_date = meeting_date
        
        if self.config.format_type == ProtocolFormat.MUNICIPAL_COUNCIL:
            return f"""Stadt {municipality}, Markt 1, {municipality}

Niederschrift
Seite 1

Stadt {municipality}
Markt 1, {municipality}

Öffentliche Niederschrift
Sitzung des {meeting_type}es

{municipality}, {formatted_date}

Gremium: {meeting_type}
Sitzung am: {formatted_date}
Sitzungsort: {municipality}, Rathaus
Sitzungsraum: Ratssaal
Sitzungsbeginn, öffentlich: 18:00 Uhr

TAGESORDNUNG

Öffentlicher Teil (Beginn: 18:00 Uhr)"""
        
        else:
            return f"""{meeting_type} - {municipality}
Datum: {formatted_date}
Automatische Transkription"""
    
    def _generate_topic_based_content(self, topics: List[TopicSegment]) -> str:
        """Generate content organized by topics"""
        content_parts = []
        
        for i, topic in enumerate(topics, 1):
            topic_parts = [f"\n{topic.title}\n"]
            
            last_speaker = None
            for segment in topic.speaker_segments:
                timestamp = self._format_timestamp(segment.start)
                
                if self.config.use_speaker_names and segment.speaker_id != last_speaker:
                    speaker_label = self._get_speaker_label(segment.speaker_id)
                    topic_parts.append(f"\n{speaker_label} {timestamp}: ")
                    last_speaker = segment.speaker_id
                elif not self.config.use_speaker_names:
                    topic_parts.append(f"{timestamp} ")
                
                topic_parts.append(f"{segment.text} ")
            
            content_parts.append("".join(topic_parts))
        
        return "\n".join(content_parts)
    
    def _generate_speaker_based_content(self, segments: List[SpeakerSegment]) -> str:
        """Generate content organized by speakers"""
        # Group segments by speaker
        speaker_segments = {}
        for segment in segments:
            if segment.speaker_id not in speaker_segments:
                speaker_segments[segment.speaker_id] = []
            speaker_segments[segment.speaker_id].append(segment)
        
        content_parts = []
        for speaker_id, speaker_segs in speaker_segments.items():
            speaker_label = self._get_speaker_label(speaker_id)
            content_parts.append(f"\n{speaker_label}:")
            
            for segment in sorted(speaker_segs, key=lambda x: x.start):
                timestamp = self._format_timestamp(segment.start)
                content_parts.append(f"{timestamp} {segment.text}")
        
        return "\n".join(content_parts)
    
    def _generate_chronological_content(self, segments: List[SpeakerSegment]) -> str:
        """Generate chronological content"""
        content_parts = []
        last_speaker = None
        
        for segment in sorted(segments, key=lambda x: x.start):
            timestamp = self._format_timestamp(segment.start)
            
            if self.config.use_speaker_names and segment.speaker_id != last_speaker:
                speaker_label = self._get_speaker_label(segment.speaker_id)
                content_parts.append(f"\n{speaker_label} {timestamp}: {segment.text}")
                last_speaker = segment.speaker_id
            else:
                content_parts.append(f"{timestamp} {segment.text}")
        
        return "\n".join(content_parts)
    
    def _generate_decisions_section(self, decisions: List[DecisionItem]) -> str:
        """Generate decisions section"""
        if not decisions:
            return ""
        
        parts = ["\nBESCHLÜSSE UND ABSTIMMUNGEN\n"]
        
        for i, decision in enumerate(decisions, 1):
            timestamp = self._format_timestamp(decision.timestamp)
            parts.append(f"{i}. {timestamp} - {decision.decision_type.upper()}")
            parts.append(f"   {decision.text}")
        
        return "\n".join(parts)
    
    def _generate_actions_section(self, action_items: List[ActionItem]) -> str:
        """Generate action items section"""
        if not action_items:
            return ""
        
        parts = ["\nAUFGABEN UND ZUSTÄNDIGKEITEN\n"]
        
        for i, action in enumerate(action_items, 1):
            timestamp = self._format_timestamp(action.timestamp)
            parts.append(f"{i}. {timestamp}")
            parts.append(f"   Aufgabe: {action.text}")
            if action.assignee:
                parts.append(f"   Zuständig: {action.assignee}")
            if action.deadline:
                parts.append(f"   Frist: {action.deadline}")
        
        return "\n".join(parts)
    
    def _generate_footer(
        self,
        municipality: str,
        meeting_date: str, 
        segments: List[SpeakerSegment],
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generate protocol footer"""
        
        try:
            formatted_date = datetime.strptime(meeting_date, "%Y-%m-%d").strftime("%d.%m.%Y")
        except ValueError:
            formatted_date = meeting_date
        
        if self.config.format_type == ProtocolFormat.MUNICIPAL_COUNCIL:
            footer_parts = [
                "\nDas Ergebnis der Beratung ergibt sich aus den Anlagen, die der Niederschrift beigefügt sind.\n",
                "Genehmigt und wie folgt unterschrieben:\n",
                f"{municipality}, den {formatted_date}\n",
                "_________________________",
                "Vorsitzende/r (z.B. Oberbürgermeister)\n",
                "_________________________", 
                "Schriftführer/in\n"
            ]
            
            if self.config.include_statistics:
                stats = self._generate_statistics(segments, metadata)
                footer_parts.insert(0, stats)
                
            return "\n".join(footer_parts)
        
        else:
            return f"\nProtokoll erstellt am: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
    
    def _generate_statistics(
        self, 
        segments: List[SpeakerSegment],
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generate protocol statistics"""
        total_duration = max(seg.end for seg in segments) if segments else 0
        total_words = sum(seg.word_count() for seg in segments)
        speakers = len(set(seg.speaker_id for seg in segments))
        
        stats_parts = [
            "\nSTATISTIKEN:",
            f"Sitzungsdauer: {total_duration/60:.0f} Minuten",
            f"Wortanzahl: {total_words}",
            f"Anzahl Sprecher: {speakers}",
            f"Wörter pro Minute: {(total_words/(total_duration/60)):.0f}" if total_duration > 0 else "Wörter pro Minute: 0"
        ]
        
        if metadata:
            if "transcription_model" in metadata:
                stats_parts.append(f"Transkriptionsmodell: {metadata['transcription_model']}")
            if "processing_time" in metadata:
                stats_parts.append(f"Verarbeitungszeit: {metadata['processing_time']:.1f}s")
        
        return "\n".join(stats_parts)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp according to configuration"""
        if self.config.timestamp_format == TimestampFormat.NONE:
            return ""
        elif self.config.timestamp_format == TimestampFormat.SIMPLE:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"({minutes:02d}:{secs:02d})"
        elif self.config.timestamp_format == TimestampFormat.DETAILED:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"({hours:02d}:{minutes:02d}:{secs:02d})"
        elif self.config.timestamp_format == TimestampFormat.RELATIVE:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"(+{minutes:02d}:{secs:02d})"
        
        return ""
    
    def _get_speaker_label(self, speaker_id: str) -> str:
        """Get formatted speaker label"""
        # This could be enhanced to use custom speaker names
        if speaker_id.startswith("SPEAKER_"):
            # Convert SPEAKER_00 to Speaker A, etc.
            try:
                num = int(speaker_id.split("_")[1])
                return f"Speaker {chr(ord('A') + num)}"
            except (IndexError, ValueError):
                pass
        
        return speaker_id


class ProtocolService:
    """
    Main protocol formatting service.
    
    Der Protokoll-Meister - verwandelt chaotische Gesprächsfetzen in
    ordentliche, offizielle Dokumente, die auch der Bürgermeister
    gerne unterschreibt.
    """
    
    def __init__(self):
        self.formatters = {}  # Cache for different configurations
    
    def format_diarized_transcription(
        self,
        diarization_result: DiarizationResult,
        municipality: str,
        meeting_date: str,
        meeting_type: str,
        config: Optional[ProtocolConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProtocolResult:
        """
        Format a diarized transcription into a protocol.
        
        Args:
            diarization_result: Result from diarization service
            municipality: Name of municipality
            meeting_date: Date of meeting (YYYY-MM-DD)
            meeting_type: Type of meeting
            config: Protocol configuration
            metadata: Additional metadata
            
        Returns:
            ProtocolResult: Formatted protocol
        """
        config = config or ProtocolConfig.get_municipal_config()
        
        # Get or create formatter
        config_key = f"{config.format_type.value}_{config.timestamp_format.value}"
        if config_key not in self.formatters:
            self.formatters[config_key] = ProtocolFormatter(config)
        
        formatter = self.formatters[config_key]
        
        return formatter.format_protocol(
            diarization_result, municipality, meeting_date, meeting_type, metadata
        )
    
    def format_simple_transcription(
        self,
        transcription_result: TranscriptionResult,
        municipality: str,
        meeting_date: str,
        meeting_type: str,
        config: Optional[ProtocolConfig] = None
    ) -> ProtocolResult:
        """
        Format a simple transcription (without diarization) into a protocol.
        
        Args:
            transcription_result: Result from transcription service
            municipality: Name of municipality  
            meeting_date: Date of meeting
            meeting_type: Type of meeting
            config: Protocol configuration
            
        Returns:
            ProtocolResult: Formatted protocol
        """
        # Convert transcription to fake diarization result
        speaker_segments = []
        for segment in transcription_result.segments:
            speaker_segment = SpeakerSegment(
                speaker_id="SPEAKER_UNKNOWN",
                start=segment.start,
                end=segment.end,
                text=segment.text,
                confidence=segment.confidence
            )
            speaker_segments.append(speaker_segment)
        
        # Create fake diarization result
        fake_diarization = DiarizationResult(
            segments=speaker_segments,
            speakers=[],  # No speaker profiles
            total_duration=transcription_result.duration,
            config=None,  # No diarization config
            processing_time=0.0
        )
        
        return self.format_diarized_transcription(
            fake_diarization, municipality, meeting_date, meeting_type, config
        )
    
    def get_format_recommendations(
        self,
        meeting_type: str,
        expected_duration_minutes: float,
        has_speakers: bool = False
    ) -> Dict[str, Any]:
        """Get format recommendations based on meeting characteristics"""
        
        recommendations = {}
        
        # Recommend format type
        if "gemeinderat" in meeting_type.lower() or "stadtrat" in meeting_type.lower():
            recommendations["format_type"] = ProtocolFormat.MUNICIPAL_COUNCIL
            recommendations["config"] = ProtocolConfig.get_municipal_config()
        elif "interview" in meeting_type.lower():
            recommendations["format_type"] = ProtocolFormat.INTERVIEW
            recommendations["config"] = ProtocolConfig.get_interview_config()
        else:
            recommendations["format_type"] = ProtocolFormat.MEETING_MINUTES
            recommendations["config"] = ProtocolConfig.get_meeting_config()
        
        # Adjust for duration
        if expected_duration_minutes > 120:  # > 2 hours
            recommendations["suggestions"] = [
                "Consider grouping by topics for long meetings",
                "Enable filler word removal for cleaner text",
                "Use simplified timestamp format"
            ]
        
        # Adjust for speaker availability
        if not has_speakers:
            recommendations["limitations"] = [
                "Speaker-specific features will be limited",
                "No speaker-based grouping available", 
                "Decisions and action items detection may be less accurate"
            ]
        
        return recommendations


# Convenience functions
def format_municipal_protocol(
    diarization_result: DiarizationResult,
    municipality: str,
    meeting_date: str,
    meeting_type: str = "Stadtrat"
) -> ProtocolResult:
    """Format municipal council protocol with standard settings"""
    service = ProtocolService()
    return service.format_diarized_transcription(
        diarization_result, municipality, meeting_date, meeting_type
    )


def format_meeting_minutes(
    diarization_result: DiarizationResult,
    municipality: str,
    meeting_date: str,
    meeting_type: str = "Meeting"
) -> ProtocolResult:
    """Format general meeting minutes"""
    service = ProtocolService()
    config = ProtocolConfig.get_meeting_config()
    return service.format_diarized_transcription(
        diarization_result, municipality, meeting_date, meeting_type, config
    )


def format_interview_transcript(
    diarization_result: DiarizationResult,
    title: str,
    date: str,
) -> ProtocolResult:
    """Return a protocol in interview format.

    Args:
        diarization_result: Output of the diarization service.
        title: Interview title (mapped to *municipality* parameter).
        date: Interview date in YYYY-MM-DD format.

    Returns:
        ProtocolResult: The fully formatted interview transcript.
    """
    service = ProtocolService()
    config = ProtocolConfig.get_interview_config()

    # Call the generic formatter with the interview-specific settings
    return service.format_diarized_transcription(
        diarization_result=diarization_result,
        municipality=title,
        meeting_date=date,
        meeting_type="Interview",
        config=config,
    )