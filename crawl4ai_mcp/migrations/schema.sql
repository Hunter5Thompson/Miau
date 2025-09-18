-- Enable required extensions
create extension if not exists vector;
create extension if not exists pg_trgm;
create extension if not exists "uuid-ossp";

-- Sources table
create table if not exists sources (
    id uuid primary key default uuid_generate_v4(),
    url text not null unique,
    title text,
    created_at timestamptz not null default now(),
    last_crawled_at timestamptz
);

-- Documents table
create table if not exists documents (
    id uuid primary key default uuid_generate_v4(),
    source_id uuid references sources(id) on delete cascade,
    url text not null,
    title text,
    md text not null,
    meta jsonb not null default '{}',
    content_hash text not null,
    fetched_at timestamptz not null default now(),
    unique (url, content_hash)
);

-- Chunks table
create table if not exists chunks (
    id uuid primary key default uuid_generate_v4(),
    document_id uuid references documents(id) on delete cascade,
    ord int not null,
    text text not null,
    heading text,
    tokens int not null,
    embedding vector(1536),
    meta jsonb not null default '{}'
);

create index if not exists idx_chunks_doc_ord on chunks(document_id, ord);
create index if not exists idx_chunks_hnsw on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_fts_idx on chunks using gin (to_tsvector('english', text));
