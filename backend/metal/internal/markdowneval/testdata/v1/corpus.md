<!-- passage:ferry-spare -->
## Spare ferry handset

Fetch a charged handset from the blue drawer beneath dispatch.
<!-- passage:rain-ingest -->
## Repeated rain packets

A roof station may retransmit an hourly packet after reconnecting. The dashboard then shows two pulses for a single shower. Compare the station identifier and the packet sequence against the previous ingestion record before reporting an unusually wet hour. Retain the first copy, discard the repeated packet, and leave both raw receipt timestamps in the incident note. A duplicate is a transport problem. Changing the gauge calibration will make the next valid reading wrong.
<!-- passage:seed-intake -->
## Seed lot intake

Record the collection meadow, date, bag count, and collector on the paper intake card. Give each bag a lot label before it enters the drying room. The nursery groups lots by source meadow, even when growth trials use the same species. If an identifier is smudged, set the bag aside with the card and ask the collector to confirm it. Do not invent a new origin from an adjacent bag's label.

The receiving shelf has space for lots awaiting identification. It is separate from the quarantine shelf for bags whose temperature history is incomplete. Enter the shelf location on the intake card before leaving the receiving bench.
<!-- passage:pump-service -->
## Silt trap servicing

Before opening the intake strainer on the west drainage pump, isolate both the electrical feed and the stored pressure. Switch off and lock the disconnect. Close the suction valve, then open the bleed and watch the pressure gauge settle at zero. When both checks are logged, lift the strainer basket.

> A silent motor does not prove the line has depressurized.

Rinse the basket into the waste tub, clean the seal groove, and replace the gasket if it will not seat flat. The second technician signs the isolation entry before either valve is reopened.
<!-- passage:tide-cache -->
## Tide board after a schedule correction

The public pier board uses a cached tide table so it can still display the next turn after a brief outage. The cache is refreshed by the overnight job. A correction sent by the hydrographer during the day can therefore appear on the office screen before it reaches the pier.

To update the pier early, compare the revision printed at the foot of the board with the revision on the office copy. If the board's copy is older, reload the signed table from the local console and check that the displayed next high water agrees with the office entry. Record the new revision in the watch log. Power cycling the display reloads the same saved copy and does not force a fresh download.

| Display condition | Action |
| --- | --- |
| Board revision older | Reload the signed table |
| Revision current, mark missing | Check the display layout |
| Link unavailable | Read the correction aloud and post the paper notice |

An older public board is a scheduling issue. A tide observer reporting that the water level itself looks wrong follows a separate procedure and compares a staff gauge reading with the sensor.
<!-- passage:rain-calibration -->
## Tipping-bucket calibration

After a field gauge dries, pour the reference cylinder into the funnel at about the rate of steady rain. Count the bucket tips over two passes. First exchange the clogged mesh if the counts are erratic. Turn the span screw only when both cleaned runs differ consistently from the reference.

Re-run the cylinder after any adjustment. Log the two trial counts, the final counts, and the cylinder's stamped volume. Mark a run invalid if the stand moved or the funnel overflowed. The gauge should stay in service when both clean counts agree with the reference.
<!-- passage:locker-tamper -->
## Locker seal inspection

The tamper seal on a trail equipment locker bridges the outside of the hinge. During the morning patrol, compare its serial to the one on the route card. Record a peeled corner or an unreadable serial as a suspected opening, even when the inventory sheet looks complete. Photograph the hinge before handling the seal. Patrol staff report this condition to maintenance and do not issue a replacement seal until the locker contents have been checked.
<!-- passage:buoy-lease -->
## Relay ownership during a buoy brownout

The north buoy sometimes loses power for a few breaths when the lamp and the telemetry radio start together. The reserve radio can take over the uplink, but both radios must never advertise the same transmitter identity at once. A neighbor can hear repeated position reports from two different shore directions and mistake the older report for a new fix.

Ownership is controlled by a time-limited relay lease kept on shore. The lamp supply does not own the lease. In particular, the power indicator coming back tells the watch that a radio can boot, but it does not say that the previous owner stopped transmitting. The watch verifies the lease record before handing the channel to another transmitter. If the record still names the original radio and its deadline is in the future, wait until the lease expires or receive a recorded surrender.

After the lease expires, ask the reserve radio to claim a new term. Read back the term and the transmitter identity. A transmit indicator without the matching term is insufficient. Disable automatic takeover when the shore clock is uncertain. A manual handoff still needs a reliable timestamp and a signed watch entry. It is safer to leave a gap in position updates than to publish two overlapping tracks.

An operator who receives a late packet should log it with its original transmit time. Do not extend a lease just because a delayed packet arrived after the deadline. The late packet can be used to repair the historical track after the live route has stabilized.

### Troubleshooting sequence

1. Compare the shore clock against the harbor time beacon.
2. Write down the current owner and the expiry time from the lease display.
3. If possible, ask the current owner to surrender. Save the message with the watch entry.
4. After expiry or recorded surrender, let the reserve radio claim the next term.
5. Verify that the reserve's reports carry the new term. Check that the previous owner's reports stop appearing as live updates.
6. Observe both radio indicators for one complete lamp cycle before returning the channel to normal monitoring.

The day crew used to reset the timer when the lamp returned. That procedure was removed after a delayed radio packet caused two tracks to overlap. A lamp budget change might help the brownouts themselves, but it cannot establish who owns the channel. Keep the electrical repair report separate from the lease log.

| Shore record | Safe response |
| --- | --- |
| Lease current and owner reachable | Request surrender and wait for acknowledgement |
| Lease current and owner silent | Wait out the recorded term |
| Lease expired on a trusted clock | Permit a new claim |
| Clock disagreeing with beacon | Hold automatic takeover and investigate |
<!-- passage:tide-surge -->
## Checking a surprising water level

A rapidly rising sea during a stiff onshore wind can put the actual water above a printed prediction. Ask the harbor observer to call out the staff gauge level and note the reading time. Compare the report with the telemetry sensor at the same minute, not with the next high-water estimate on the public tide board.

Use the surge notice form when both measurements agree and exceed the prediction. The notice should identify the affected stairways and pontoons. Do not revise the astronomical schedule to make it fit one storm. If the readings disagree, check the staff gauge reference mark and look for debris against the pressure sensor.
<!-- passage:weather-replay -->
## Replaying a missing marsh weather interval

The marsh station records a local minute series even during a radio outage. A missing interval on the shore graph can sometimes be repaired from that recorder. The purpose of replay is to fill missing time slots without changing a measurement already accepted for a slot. Replayed data also needs a visible provenance trail so a report can tell it from the live feed.

Start by identifying the station and time basis. The recorder's export names the station, the local clock offset, and the first and last samples carried on the card. A file copied from the storm pole looks similar but uses a different station key. Confirm the station identifier on the plate and in the file. Never infer it from the filename. If the station clock moved while the radio was down, split the export at the adjustment and resolve the time mapping for each portion before import.

Make a dry-run ledger of requested timestamps. For each sample, check whether the shore recorder already has an accepted value for the same station and minute. Omit an identical sample. If a live sample exists with a different value, keep the accepted value and put the replay sample into a conflict report. If a gap is marked as intentionally suppressed because of a safety or maintenance closure, leave the slot suppressed. Only timestamps still marked missing receive replay records.

The first import must use a scratch copy of the interval. Compare the count of newly filled slots with the dry-run ledger. Check the first and last timestamps and inspect the minute on each side. Difference the scratch records against the accepted set and confirm that all prior values are intact. If the export straddles midnight, check both date partitions individually. A timezone mistake can produce values that look reasonable in the plot while being attached to the wrong day.

Use the import summary to record the original card identifier, the interval, the authorizing watch officer, the proposed count, and the actual inserted count. Record conflicts with both values and their origins. Preserve the unmodified card export separately. If a second replay proposes zero inserts, record that result rather than deleting the first import summary.

### Example after a transmitter outage

An export covers 05:58 through 06:07. The live stream has values for 05:58, 05:59, and 06:07. The outage record marks 06:04 suppressed while a technician disconnected the wind vane. The import should fill 06:00 through 06:03 and 06:05 through 06:06. A conflicting sample at 06:07 is reviewed in the conflict report. Leave the accepted 06:07 value and the 06:04 suppression as they were.

### Clock correction across a missing interval

A maintenance ticket sometimes records the time a technician changed the station clock but omits the magnitude of the change. In that case identify at least two samples on each side that can be matched to a known event. The wind mast has a local service switch that marks its state changes in the minute export. The duty log for the same switch is on the shore clock. Use those marks to bound the offset before and after the correction. A resemblance between two gust peaks is useful as a clue, but it is too ambiguous to serve as the only time reference.

When the clock is set back, two source rows may have the same local timestamp. Keep their original file offsets and disambiguate them by the service-switch marks. Never choose the later one merely because it appears lower in a spreadsheet. If there is no satisfactory time mapping, import the unaffected portions only. Mark the unresolved segment in the review request and include the relevant switch log pages so the next watch can resume the investigation.

### Card changes and recovery notes

Two cards can each contain part of the same outage. Export each card independently and give each export its own identifier. Join records in the dry-run ledger only after translating both time bases. A card from a second station can supplement a missing interval for that station, but its measurements cannot be substituted for the original station in a marsh report. Gaps between cards are expected when a technician waits for a dry break before opening the recorder housing.

If an import was interrupted, consult the saved summary before attempting to resume. Test the same proposed timestamps against the current accepted set. The part committed before the interruption should now be omitted, and the remaining eligible gaps should still be proposed. Record a fresh summary even if the second attempt inserts nothing. This makes the history readable without relying on the shell transcript. At the end of the watch, hand over the unresolved conflicts and uncertain timestamps separately. A plausible looking chart alone is not enough to close the outage.

The replay process repairs storage at shore. It does not calibrate a rain gauge, repair the radio, or change the station clock. Feed monitor alerts can be cleared only after new live samples arrive with fresh transmit times.
<!-- passage:seed-hold -->
## Warming event in the seed store

When a sealed seed lot may have been above the cooler limit for an unknown duration, move the container to the quarantine shelf and mark it held. Attach the last known good timestamp and the first observed warm reading to the lot record. Do not send it back to the shelf for planting just because the replacement cooler is now cold.

The nursery coordinator reviews the recorder trace and the lot's expected tolerance. If the event cannot be bounded, request a viability sample and wait for a written disposition. A cleared lot returns to active inventory with its original identifier and the release decision on the same card. A failed lot stays held until the coordinator arranges disposal.
<!-- passage:pump-display -->
## A frozen drainage dashboard

The drainage office chart occasionally stops drawing new motor readings after the display sleeps. Check the time beside the last displayed point. If the controller continues to advance its local counter, reload just the dashboard and check the chart again. Save a screenshot for the office ticket.

A stalled chart is not authority to open a pump. Call field operations if both the dashboard and the local readings stop. Keep the emergency control procedure next to the console and use the usual on-site isolation checks for work on the equipment.
<!-- passage:locker-release -->
## Releasing a rescued equipment locker

When the emergency key opens a trail equipment locker for a rescue call, the dispatcher records the time, the key number, and who took equipment. The locker stays reserved until a returning crew reconciles the kit inventory. The crew notes which pieces came back, which went to repair, and which were consumed.

After the return, a second person compares the inventory with the dispatch record and signs the release line. Only then may dispatch clear the reservation and offer the kit to the next crew. In a second immediate emergency, dispatch issues a different kit. A new seal is evidence of a closed door, not a substitute for the signed reconciliation.
<!-- passage:ferry-restock -->
## Restocking the ferry radio drawer

At the end of the swing shift, wipe returned radio handsets and put them in the charger cradle. Put them into the spare drawer once the steady light on the cradle shows a full charge. A radio with a damaged battery latch goes into the repair bin at the end of the counter.

Write the serials from the two functioning spares on the evening inventory card. If the count drops below two, report a supply issue to the duty officer. The restock entry shows who replaced missing spares and when. During a crossing the dispatch worker should use the spare procedure instead of taking a radio apart at the desk.
<!-- passage:buoy-scale -->
## Converting buoy pressure packets

Each pressure packet from the harbor buoy has a raw reading and a scale code. Older transmitters use a different conversion from newer ones, even where the onboard windows display the same engineering units. Before using a packet in the water-level chart, look up its scale code in the equipment register. Use that row's gain and offset, and store the original raw reading with the derived value.

If the scale code is unknown, put the packet on the decoding hold queue. A technician must compare the instrument plate with the most recent calibration sheet before adding a conversion row. A guessed scale can turn an ordinary tide into a plausible but false surge. The hold is cleared by publishing a new row with the approved instrument record attached.

```text
reading = raw * gain + offset
key = (instrument_serial, scale_code)
```

A new transmitter can use the same uplink identity after the watch grants it a lease. That transport identity is not a scale code. Check both values when an instrument is replaced. The decoding hold queue is for packets whose measurement cannot be interpreted, while the lease record controls which radio may publish live packets.
