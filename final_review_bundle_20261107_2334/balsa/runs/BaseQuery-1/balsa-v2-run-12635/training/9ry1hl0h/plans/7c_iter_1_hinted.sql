/*+ MergeJoin(an pi ci lt t ml n it)
 HashJoin(pi ci lt t ml n it)
 HashJoin(pi ci lt t ml n)
 HashJoin(ci lt t ml n)
 HashJoin(ci lt t ml)
 HashJoin(lt t ml)
 HashJoin(t ml)
 IndexScan(an)
 SeqScan(pi)
 SeqScan(ci)
 SeqScan(lt)
 SeqScan(t)
 IndexScan(ml)
 IndexScan(n)
 SeqScan(it)
 Leading((an ((pi ((ci (lt (t ml))) n)) it))) */
SELECT MIN(n.name) AS cast_member_name,
       MIN(pi.info) AS cast_member_info
FROM aka_name AS an,
     cast_info AS ci,
     info_type AS it,
     link_type AS lt,
     movie_link AS ml,
     name AS n,
     person_info AS pi,
     title AS t
WHERE an.name IS NOT NULL
  AND (an.name LIKE '%a%'
       OR an.name LIKE 'A%')
  AND it.info ='mini biography'
  AND lt.link IN ('references',
                  'referenced in',
                  'features',
                  'featured in')
  AND n.name_pcode_cf BETWEEN 'A' AND 'F'
  AND (n.gender='m'
       OR (n.gender = 'f'
           AND n.name LIKE 'A%'))
  AND pi.note IS NOT NULL
  AND t.production_year BETWEEN 1980 AND 2010
  AND n.id = an.person_id
  AND n.id = pi.person_id
  AND ci.person_id = n.id
  AND t.id = ci.movie_id
  AND ml.linked_movie_id = t.id
  AND lt.id = ml.link_type_id
  AND it.id = pi.info_type_id
  AND pi.person_id = an.person_id
  AND pi.person_id = ci.person_id
  AND an.person_id = ci.person_id
  AND ci.movie_id = ml.linked_movie_id;

