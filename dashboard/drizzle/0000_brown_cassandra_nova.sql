CREATE TABLE `paper_annotations` (
	`id` text PRIMARY KEY NOT NULL,
	`owner_key` text NOT NULL,
	`document_id` text NOT NULL,
	`section_id` text NOT NULL,
	`section_title` text NOT NULL,
	`selected_text` text NOT NULL,
	`start_offset` integer NOT NULL,
	`end_offset` integer NOT NULL,
	`prefix` text DEFAULT '' NOT NULL,
	`suffix` text DEFAULT '' NOT NULL,
	`comment` text NOT NULL,
	`status` text DEFAULT 'open' NOT NULL,
	`created_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL,
	`updated_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL
);
--> statement-breakpoint
CREATE INDEX `idx_paper_annotations_owner_document_created` ON `paper_annotations` (`owner_key`,`document_id`,`created_at`);