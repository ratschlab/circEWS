library(ggplot2)
options(bitmapType='cairo')
#summary_dir <- '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/3a_endpoints/180110/summary/'
#summary_dir <- '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/3a_endpoints/180214/summary/'
#summary_dir <- '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/3a_endpoints/180704/reduced/summary/'
#summary_dir <- '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/3a_endpoints/v6b/reduced/summary/'
#summary_dir <- '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/endpoints/180822/reduced/summary/'
#summary_dir <- '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/endpoints/180926/reduced/summary/'
#summary_dir <- '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/endpoints/181023/reduced/summary/'
summary_dir <- '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/endpoints/181103/reduced/summary/'
#summary_dir <- '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/endpoints/v6b/reduced/summary/'

summary <- read.csv(paste0(summary_dir, 'summary.csv'))
summary <- subset(summary, event %in% c('event1', 'event2', 'event3'))

number_of_events = subset(summary, variable=="number_of_events")
ggplot(number_of_events, aes(x=value, fill=event)) + geom_histogram() + facet_wrap(~event, nrow=3, scales='free_y') + scale_y_log10() + xlab("number of events") + ylab("number of patients") + guides(fill=FALSE) + ggtitle("number of events")
ggsave(paste0(summary_dir, "number_of_events.pdf"))
ggsave(paste0(summary_dir, "number_of_events.png"), width=4, height=5, type='cairo')


average_duration = subset(summary, variable=="average_duration")
ggplot(average_duration, aes(x=value/60, fill=event)) + geom_histogram() + facet_wrap(~event, nrow=3, scales='free_y') + xlab("average event duration (hours)") + ylab("number of patients") + guides(fill=FALSE) + ggtitle("average event duration")
ggsave(paste0(summary_dir, "average_duration.pdf"))
ggsave(paste0(summary_dir, "average_duration.png"), width=4, height=5)

#range_of_durations = subset(summary, variable=="range_of_durations")
#ggplot(range_of_durations, aes(x=value/60, fill=event)) + geom_histogram() + facet_wrap(~event, nrow=3, scales='free') + xlab("range of event durations (hours)") + ylab("number# of patients") + guides(fill=FALSE)
#ggsave("range_of_durations.pdf")
#ggsave("range_of_durations.png")

time_to_first_event = subset(summary, variable=="time_to_first_event")
ggplot(time_to_first_event, aes(x=value/60, fill=event)) + geom_histogram() + facet_wrap(~event, nrow=3, scales='free_y') + scale_x_log10() + xlab("time to first event (hours)") + ylab("number of patients") + guides(fill=FALSE) + ggtitle("time to first event")
ggsave(paste0(summary_dir, "time_to_first_event.pdf"))
ggsave(paste0(summary_dir, "time_to_first_event.png"), width=4, height=5)


fraction_in_event = subset(summary, variable=="fraction_in_event")
ggplot(fraction_in_event, aes(x=value, fill=event)) + geom_histogram() + facet_wrap(~event, ncol=3, scales='free') + xlab("fraction of time in event (excluding 0)") + ylab("number of patients") + guides(fill=FALSE) + xlim(1e-5, 1) + ggtitle("fraction of time in event")
ggsave(paste0(summary_dir, "fraction_in_event.pdf"))
ggsave(paste0(summary_dir, "fraction_in_event.png"))

 
# --- now get aggregate statistics --- #
summary <- read.csv(paste0(summary_dir, 'summary.csv'))
length_of_stay = subset(summary, variable=="length_of_stay")
ggplot(length_of_stay, aes(x=value/(24*60))) + geom_histogram() + scale_x_log10() + xlab("length of stay (days)") + ylab("number of patients") + ggtitle("length of stay")
ggsave(paste0(summary_dir, "length_of_stay.pdf"))
ggsave(paste0(summary_dir, "length_of_stay.png"))

number_of_deteriorations = subset(summary, variable=="number_of_deteriorations")
perc_zero <- mean(number_of_deteriorations$value == 0)*100
ggplot(number_of_deteriorations, aes(x=value)) + geom_histogram() + xlab(paste0("number of deteriorations (excluding 0); ", perc_zero, "% of patients have 0")) + scale_x_log10() + ylab("number of patients") + ggtitle("number of deteriorations")
ggsave(paste0(summary_dir, "number_of_deteriorations.pdf"))
ggsave(paste0(summary_dir, "number_of_deteriorations.png"))
